# ADR 0029: Close block LSQ and select full-Gram S8

Date: 2026-07-11

## Status

Accepted as a quantizer rejection and one final bounded activation-aware
successor. No teacher-forced, token, context-ladder, or product promotion is
implied.

## Context

Clean seq682 applies ADR 0028's fixed K32-block activation-LSQ rounding. The
generated runtime core remains fast at `3015.947 / 3012.896 us`. The offline
optimizer changes `29,808,169` of `268,435,456` codes (`11.1044%`), converging
in `2.858` mean sweeps (`12` maximum).

Activation weighting produces real but insufficient movement. Output relative
L2 improves from seq677's `0.00500705` to `0.00381165` (`23.87%` reduction),
yet remains `1.906x` above the `0.002` contract. The block-diagonal objective
cannot compensate correlated error across K32 groups.

Evidence:

- `output/onednn-grouped-q6-activation-lsq-gate-20260711Tseq682cleanZ/`
- `output/onednn-grouped-q6-s8-per32-gate-20260711Tseq677cleanZ/`

## Decision

Close K32-block-diagonal activation-LSQ rounding. Select exactly one stronger
offline objective while retaining the same runtime payload and core: form one
shared `512 x 512` Gram matrix from all 8192 reconstructed runtime source rows,
then run the same strict-improvement discrete coordinate descent jointly over
all 512 codes of each expert/output row. Each code continues to use its fixed
max-absolute K32 scale and `[-127, 127]` range.

No damping, per-expert Gram, calibration subset, iteration cap, clipping,
scale, group, ordering, or coordinate variant is authorized. The layer-39 core
must remain `<=4316.404 us` and pass all `16,777,216` outputs at finite, cosine
`>=0.999`, relative L2 `<=0.002` in primary and confirm.

## Consequences

- Block-local activation-aware rounding is closed despite its measured
  improvement.
- Full-Gram rounding is the final activation-calibrated S8-per-K32 attempt. A
  miss closes this quantization family; do not proceed to per-expert overfit or
  calibration-set variants.
- Any pass still advances through all-layer and teacher-forced overfit gates.

## Follow-Up

- Run one paired full-Gram activation-LSQ gate on layer 39.
- If it passes, apply the identical algorithm to all Q6 layers. If it fails,
  record movement and change representation/architecture family.

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
