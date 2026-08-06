# ADR 0021: Reopen grouped-prefill component accuracy

Date: 2026-07-11

## Status

Superseded by ADR 0022. Seq673 closes the Q4 side with an F32 SwiGLU handoff;
the Q6 accuracy/performance route remains open under a new fixed codegen gate.

## Context

The locked acceptance matrix requires every component output to be finite,
have cosine similarity at least `0.999`, and have relative L2 at most `0.002`.
The earlier grouped-prefill gates checked finiteness and an absolute `5e-3`
outlier threshold, but did not enforce relative L2. That omission allowed low-
magnitude systematic error to appear correct.

Clean seq671 captures six routed-MoE boundaries for every layer from one live
1024-token CPU model evaluation (`240` tensors), loads all 40 real payloads in
one native context, and compares every native layer output. SwiGLU passes the
contract in all 40 layers. Down and routed output do not:

| codec | layers passing weighted down | weighted-down relL2 range | layers passing routed output | routed-output relL2 range |
|---|---:|---:|---:|---:|
| Q4_K | 0 / 20 | `0.002337..0.002735` | 0 / 20 | `0.002251..0.002637` |
| Q6_K centered-U8 surrogate | 0 / 20 | `0.004596..0.006237` | 0 / 20 | `0.004445..0.006066` |

Aggregate weighted-down relative L2 is `0.00532031`; aggregate routed-output
relative L2 is `0.00508737`. Cosine remains above `0.999`, outputs are finite,
and all 40 real layer computations execute, so this is a representation/input-
quantization accuracy failure rather than a load, scheduling, or NaN failure.

Evidence:

- `output/all-layer-mixed-component-20260711Tseq671cleanZ/result.json`
- `benchmarks/intel-qwen36-35b-a3b-gguf-q4km/acceptance-matrix.json`

## Decision

Reopen component accuracy before live-state chaining. Treat seq669 only as a
mixed-schedule timing lower bound and seq670 only as a residency/capacity proof.
Neither the current grouped Q4 carrier nor the centered-U8 Q6 surrogate is an
accepted correctness carrier.

Update all grouped component runners and formal gates to enforce the locked
cosine, relative-L2, and finite-output contract. Select one measured exact-Q6_K
per-16 feasibility probe on the worst observed layer before considering any
representation sweep. Q4 source quantization must also be repaired under the
same contract.

This decision does not change the hard `1.10x` same-run OpenVINO product target,
does not reject the resident O(1)-in-layer-count architecture, and does not
authorize teacher-forced, token, context, or product performance claims.

## Consequences

- No live preceding-state integration may proceed until all 40 layer components
  meet relative L2 `<=0.002`.
- Absolute-error-only component gates are invalid for promotion.
- The seq669 timing sum remains useful only as a kill-number for successor
  representations; successor timing must be remeasured if its payload or work
  count changes.
- Materialized-F16, NPU, oneDNN/OpenVINO runtime linkage, direct raw-row DP4A,
  and the old `1.25x` target remain closed.

## Follow-Up

- Measure exact Q6_K per-16 accuracy and complete layer time on the worst seq671
  Q6 layer.
- Reduce Q4 and Q6 weighted-down and routed-output relative L2 to `<=0.002`, then
  rerun the all-40 component gate before live-state chaining.

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
