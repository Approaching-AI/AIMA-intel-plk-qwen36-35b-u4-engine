# ADR 0037: Reject Q4 F32 and select a teacher-forced state ladder

Date: 2026-07-11

## Status

Accepted as a Q4-F32 component capability and rejected as the mixed token-state
repair. Product correctness and speed remain open.

## Context

ADR 0036 selected an F32 Q4 down destination because Q4-F16 and Q6-F32 each
passed the fixed eight-token row alone while their combination failed.

Clean seq694 proves pinned offline generation of a distinct Q4 F32 down binary
(`8c593d290c58ede7176ed571e5a47e3a4a220f79183c76ca84f113655c78dcd7`).
Clean seq695 routes it through an explicit F32 contribution plane and passes
all 40 component rows. Aggregate relative L2 is `0.000209339` for SwiGLU,
`0.000317829` for weighted down, and `0.000304693` for routed output over
`167,772,160`, `671,088,640`, and `83,886,080` values.

The token-state result rejects the repair:

- clean seq696 applies Q4-F32 plus Q6-F32 and diverges at zero-based token 2;
  baseline IDs are `[264,264,271,248068,198,8160,579,264]`, while the injected
  row emits eight `264`s;
- clean seq697 applies Q4-F32 alone and produces the same failure, despite a
  passing first distribution (`KLD 0.000229606`, top-1 token `264`) and lower
  same-state aggregate routed relative L2 `0.000150329`;
- the accepted Q4-F16-only seq689 row remains exact, so removing F16 rounding
  is not a causal token-state improvement.

Aggregate component error and the first predicted distribution are therefore
insufficient to choose the next precision boundary. Greedy continuation also
changes the input sequence after the first divergent ID, hiding where the
shared-prefix distribution first becomes unstable.

Evidence:

- `output/grouped-q4-f32-down-codegen-20260711Tseq694cleanZ/`
- `output/all-layer-mixed-f32-component-20260711Tseq695cleanZ/`
- `output/all-layer-live-state-mixed-f32-tokens-20260711Tseq696cleanZ/`
- `output/all-layer-live-state-q4-f32-tokens-20260711Tseq697cleanZ/`

## Decision

Keep the Q4 F32 generator/runtime mode as diagnostic capability, but do not use
it as the selected R2 carrier. Restore Q4-F16 plus Q6-F32 as the attribution
carrier.

Before changing source quantization, weights, schedules, layers, or kernels,
add one teacher-forced continuation mode. Feed the fixed baseline IDs to both
baseline and injected contexts and record the full-vocabulary distribution at
every shared-prefix position. Require finite logits and report KLD, cosine,
relative L2, top-1 equality, top-1 margin, and the first failing position.

This ladder is attribution evidence, not an acceptance relaxation: final
promotion still requires the original greedy IDs to match exactly.

## Consequences

- Q4 F32 cannot be inferred correct from lower aggregate relative L2.
- Q6 F32 remains accepted in isolation by seq692.
- Layer masks, interpolation coefficients, prompt changes, and threshold
  tuning are not admitted before the shared-prefix ladder identifies the first
  unstable distribution.
- No speed claim is allowed from seq694--697.

## Follow-Up

- Extend the existing live-injection harness with fixed continuation tokens
  and per-position full-logit artifacts.
- Run one paired Q4-F16/Q6-F32 teacher-forced row on the locked 1024-token
  prefix, then choose the next precision boundary from that evidence.

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
