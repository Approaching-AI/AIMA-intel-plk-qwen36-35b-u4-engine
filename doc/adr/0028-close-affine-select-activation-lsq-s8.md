# ADR 0028: Close affine correction and select activation-LSQ S8

Date: 2026-07-11

## Status

Accepted as a representation/timing rejection and one bounded offline quantizer
successor. No teacher-forced, token, context-ladder, or product promotion is
implied.

## Context

Clean seq681 proves ADR 0027's external affine decomposition is implemented as
specified. GPU matches the host affine calculation within `6.131e-8`; direct
oneDNN zero points are not needed for correctness. The fixed min/max affine
codec still fails the component contract: weight relative L2 is `0.00474459`,
and both full output comparisons have cosine `0.999990224` and relative L2
`0.00442190`.

The compact correction also leaves no timing case to optimize. Main plus source
group sums plus correction is `4701.548 / 4701.153 us`, about `385 us` above
the `4316.404 us` gate. Both axes fail, so workgroup or correction-kernel tuning
would optimize an inaccurate representation.

The measured fast carrier remains zero-point-free S8-per-K32 at roughly 3.0 ms.
Its failure is the offline nearest/max-absolute rounding rule, not generated
core execution. The next materially different question is whether one fixed
activation-weighted rounding rule can use the same runtime payload and core.

Evidence:

- `output/onednn-grouped-q6-external-affine-zp-gate-20260711Tseq681cleanZ/`
- `output/onednn-grouped-q6-s8-per32-gate-20260711Tseq677cleanZ/`

## Decision

Close min/max affine per-K32 quantization and its external zero-point
correction. Select exactly one offline activation-LSQ S8-per-K32 quantizer on
layer 39:

1. reconstruct the accepted runtime S8 source values for all 8192 captured
   assignments;
2. form one shared `32 x 32` Gram matrix for each of the 16 K32 groups;
3. keep the max-absolute F32 scale per expert/output/K32;
4. start from nearest S8 codes and run deterministic discrete coordinate
   descent on `e^T H e`, accepting only strict objective reductions until no
   code changes; clamp codes to `[-127, 127]`.

The Gram matrices are shared across experts rather than fitted per expert, and
no damping, threshold, iteration count, clipping, scale, group, or calibration
variant is authorized. The resulting single generated core must pass all
`16,777,216` outputs at finite, cosine `>=0.999`, relative L2 `<=0.002`, and
remain `<=4316.404 us` in primary and confirm.

## Consequences

- Affine U8, external zero-point correction, and correction-kernel tuning are
  closed.
- The runtime representation remains the already measured S8/F32-scale K32
  payload; only offline code selection changes.
- A layer-39 pass is still only a component preflight. All-layer and
  teacher-forced gates must detect calibration overfit before promotion.

## Follow-Up

- Run the fixed activation-LSQ layer-39 paired gate once.
- If it passes, apply the same parameter-free algorithm to all Q6 layers and
  rerun the all-40 component gate. If it fails, record the achieved movement
  before changing representation family.

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
