# ADR 0030: Close full-Gram S8 and select the correct R2 carrier

Date: 2026-07-11

## Status

Accepted as a quantizer-family closure and a correctness-route switch. The
selected exact-Q6 implementation is an R2 correctness carrier only; its
measured performance failure remains in force and no speed, token, or product
promotion is implied.

## Context

Clean seq683 runs ADR 0029's final activation-calibrated S8-per-K32 attempt on
worst Q6 layer 39. One shared `512 x 512` Gram matrix drives full-row discrete
coordinate descent over every output row. The optimizer changes `38,400,669`
of `268,435,456` codes (`14.3054%`) in `9.061` mean sweeps (`32` maximum).

The fixed generated core remains below its raw cap at
`3015.195 / 3012.663 us`, and the GPU matches the repacked representation.
Full-Gram rounding lowers output relative L2 from seq677's `0.00500705` to
`0.002641742`, a `47.26%` reduction. It nevertheless misses the mandatory
`0.002` contract by `1.321x` in both `16,777,216`-value comparisons.

All preregistered fast K32 S8 variants are now closed: nearest, dense and
sparse residual, affine zero-point, block-Gram, and full-Gram. Continuing with
calibration, damping, scale, or sweep variants would be an unbounded codec
sweep. Separately, clean seq675 already supplies a correct exact-Q6 per-16/F32
path at relative L2 `0.000206724`, although its handwritten M16 mapping is too
slow for product promotion (`6630.000 / 6629.895 us`).

The factory roadmap defines R2 as the complete slow engine that emits correct
tokens and becomes the speed denominator. The project has not yet crossed that
gate.

Evidence:

- `output/onednn-grouped-q6-full-gram-lsq-gate-20260711Tseq683cleanZ/`
- `output/q6-exact-per16-prefill-gate-20260711Tseq675cleanZ/`
- `meta-engine-factory/doc/methodology/00-engine-and-roadmap.md`

## Decision

Close activation-calibrated S8-per-K32 quantization. Switch from representation
search to R2 completion using the already-correct carriers:

- Q4 layers use seq673's auxiliary-F32 SwiGLU path;
- Q6 layers preserve every signed Q6 value and F32 per-16 scale, and consume
  the same F32 SwiGLU boundary;
- the parameterized resident runtime owns all 40 real layer payloads in one
  native OpenCL context and maps no oneDNN/OpenVINO runtime library.

The next gate is an all-40 component rerun from one live 1024-token capture.
Every layer and all aggregate SwiGLU, weighted-down, and routed-output values
must be finite, cosine `>=0.999`, and relative L2 `<=0.002`. Timing is recorded
but is not a promotion gate for this R2 confirmation.

No new quantizer, kernel shape, workgroup, tile, scale representation, or
calibration variant is authorized during this integration. Once all-40
components pass, chain the same runtime through live transformer state and run
teacher-forced distribution, deterministic-token, long-context sentinel, and
context-ladder correctness before reopening profile-backed optimization.

## Consequences

- Full-Gram S8 is rejected despite a material accuracy improvement and a fast
  core; the representation still fails the contract.
- The exact-Q6 M16 performance rejection remains valid. Selecting it for R2
  correctness does not reopen it as the final speed carrier.
- Product performance remains failed. The hard target is still per bucket and
  phase `max(absolute target, 1.10x same-run OpenVINO)`.
- Future Q6 performance work starts from a correct end-to-end engine profile,
  not another offline codec variant.

## Follow-Up

- Add exact-per-16 Q6 payloads and execution to the parameterized resident
  grouped-prefill runtime.
- Run one clean all-40 component gate from the existing live-capture protocol.
- On a pass, integrate that carrier into the live-state loop and advance the
  correctness ladder.

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
