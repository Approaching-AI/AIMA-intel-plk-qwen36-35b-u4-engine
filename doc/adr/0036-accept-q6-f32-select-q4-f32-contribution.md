# ADR 0036: Accept Q6 F32 and select Q4 F32 contribution

Date: 2026-07-11

## Status

Accepted as a Q6 precision repair and a mixed-state rejection with one
symmetrical Q4 successor. No product correctness or speed promotion is implied.

## Context

Clean seq692 applies only the 20 exact-Q6 layers after ADR 0035's F32
contribution repair. It now passes the complete fixed gate:

- aggregate same-state routed relative L2 `0.000186787`;
- first full-vocabulary KLD `0.000298060`, top-1 token `264` in both rows;
- exact continuation
  `[264,264,271,248068,198,8160,579,264]` in both rows.

The old Q6-only F16 row seq690 diverged at position 2, so the F32 store/scatter
change is a causal correctness repair.

Clean seq693 then applies all 40 layers with Q6 F32 and the existing Q4 F16
contribution path. Components and first distribution still pass—aggregate
relative L2 `0.000224768`, KLD `0.000643465`, first token `264`—but exact
continuation again diverges at position 2. Candidate IDs are
`[264,264,4108,25,35494,18,21,829]`.

Seq689 proves Q4 F16 passes alone; seq692 proves Q6 F32 passes alone; seq693
proves their combination fails. The remaining common additive precision loss
is Q4 down's generated F32 accumulator conversion to F16 contribution before
scatter.

Evidence:

- `output/all-layer-live-state-q6-f32-tokens-20260711Tseq692cleanZ/`
- `output/all-layer-live-state-mixed-f32-tokens-20260711Tseq693cleanZ/`
- `output/all-layer-live-state-q4-attribution-20260711Tseq689cleanZ/`

## Decision

Accept exact-Q6 F32 contribution/store/scatter for R2 correctness and reject
the mixed Q6-F32/Q4-F16 token state.

Add one offline-generated Q4 down binary whose grouped destination is F32.
Route its weighted contribution plane through the same deterministic F32
scatter used by exact Q6. Keep Q4 gate/up, Q4 codes/scales/min repair, F32
SwiGLU, source Q8 quantization, schedule, workgroup, layer set, and prompt
unchanged.

First prove the generator emits and the all-40 component gate passes. Then run
the mixed eight-token row; require the same component, KLD `<=0.005`, top-1,
and exact ID gates.

## Consequences

- Q6 source-Q8 precision is not reopened; its F32 carrier is token exact in
  isolation.
- Q4 F32 output is an R2 correctness carrier. Increased traffic and timing are
  recorded but cannot be promoted as product performance.
- A mixed pass advances to broader prompt/context correctness. A miss requires
  additive error attribution before moving source quantization.

## Follow-Up

- Add a guarded F32 destination mode to the offline Q4 grouped generator.
- Generate one Q4-down F32 binary, integrate an explicit Q4-F32 layer kind,
  and rerun all-layer components plus mixed tokens.

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
