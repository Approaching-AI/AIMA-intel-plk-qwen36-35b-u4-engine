# ADR 0026: Close ternary sparsity and select affine U8 per-32

Date: 2026-07-11

## Status

Accepted as a feasibility rejection and one bounded representation successor.
No teacher-forced, token, context-ladder, or product promotion is implied.

## Context

Clean seq679 runs ADR 0025's deterministic least-squares ternary residual
census. Its dense reference is accurate enough: corrected weight relative L2
is `0.00176792`, and both full output comparisons pass at cosine `0.999998656`
and relative L2 `0.00164043`.

The preregistered sparsity gate fails. The optimal ternary codec retains
`171,845,315 / 268,435,456` nonzero weights, density `0.640174`, versus the
zero-overhead ceiling `0.423513`. Even linear scaling from seq677's S8 core
would consume about `1941 us` before sparse indices, scales, accumulation, or
the F32 add, already above the `1284.197 us` residual budget. No sparse kernel
is admitted.

Evidence:

- `output/onednn-grouped-q6-s8-ternary-residual-census-20260711Tseq679cleanZ/`
- `output/onednn-grouped-q6-s8-s4-residual-gate-20260711Tseq678cleanZ/`

## Decision

Close ternary/sparse residual correction under the current S8 main plane.
Select exactly one single-core representation that can address per-K32
asymmetry without a second matrix: affine U8 quantization of each dequantized
Q6_K K32 group, using the group's min/max-derived F32 scale and one U8 zero
point. The group size is fixed at 32, matching the measured DPAS granularity.

On layer 39 the generated core must compare all `16,777,216` values at finite,
cosine `>=0.999`, relative L2 `<=0.002`, and remain `<=4316.404 us`. Failure
closes affine per-K32 quantization; do not vary clipping, calibration, zero-point
type, group size, or JIT strategy.

This does not reopen the crashing exact-per-16 U8 zero-point route: that route
uses unsupported K16 quantization groups, while this gate is K32-aligned.

## Consequences

- Optimal ternary is an accuracy reference only; its `0.640174` density is
  arithmetically too high for a sparse implementation.
- Dense or sparse second residual matrices remain closed.
- A passing affine-U8 preflight would still require binary extraction and the
  pure-OpenCL complete-boundary gate.

## Follow-Up

- Run one paired affine-U8-per-K32 all-value/raw-core gate on layer 39.
- If it fails, record accuracy and timing separately before changing codec
  family; do not sweep affine quantizers.

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
