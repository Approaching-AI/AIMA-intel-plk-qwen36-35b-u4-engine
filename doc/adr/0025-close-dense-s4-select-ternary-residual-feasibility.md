# ADR 0025: Close dense S4 and select ternary-residual feasibility

Date: 2026-07-11

## Status

Accepted as a component-route rejection and a bounded codec feasibility gate.
No runtime implementation or product promotion is implied.

## Context

Clean seq678 executes ADR 0024's fixed S8 main plus signed-S4 residual. The
codec is comfortably correct: residual-corrected weight relative L2 is
`0.000370624`, and both full output comparisons reach cosine `0.999999938` and
relative L2 `0.000351474`.

The dense second matrix is terminally over budget. Main plus residual plus the
native F32 add takes `7381.741 / 7353.581 us` versus `4316.404 us`, a
`3065.337 / 3037.177 us` miss. Relative to seq677's slower `3032.207 us` main,
the residual composition costs about `4321.374 us` but has only `1284.197 us`
available; it needs at least `3.365x` residual-side movement. The S4 codec is
also about `5.4x` more accurate than the output contract requires, so the next
question is whether a much sparser correction exists, not another dense
low-bit GEMM.

Evidence:

- `output/onednn-grouped-q6-s8-s4-residual-gate-20260711Tseq678cleanZ/`
- `output/onednn-grouped-q6-s8-per32-gate-20260711Tseq677cleanZ/`

## Decision

Close a dense second residual matrix, including the measured signed-S4 form.
Before implementing another kernel, run one offline optimal-ternary residual
census on layer 39. For each K32 residual group, deterministically choose the
subset of largest absolute values and its least-squares shared magnitude; the
selected subset size is the one minimizing that group's total squared error.
Codes are fixed to `{-1, 0, +1}`.

The census must answer both gates:

1. dense-reference output over all `16,777,216` values is finite, cosine
   `>=0.999`, and relative L2 `<=0.002`;
2. nonzero density is below the zero-overhead hard ceiling
   `1284.197 / 3032.207 = 0.423513`.

If either fails, do not implement a sparse kernel. If both pass, derive and run
one sparse native correction under the unchanged combined `4316.404 us` gate.
The density ratio is only an admission bound, not a speed claim.

## Consequences

- Dense S4, dense S8 residual, and another dense grouped residual primitive are
  closed; their full-matrix work cannot fit the measured residual budget.
- The ternary census is an offline codec/reference calculation. It may use a
  dense reference primitive solely to establish output accuracy, but that
  primitive is not a timing carrier.
- No threshold, bit width, group size, scale rule, or tile sweep is authorized.

## Follow-Up

- Record ternary corrected weight error, nonzero density, and all-value output
  accuracy once on layer 39.
- Implement a sparse correction only if both preregistered feasibility gates
  pass; otherwise record the terminal result and change representation family.

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
