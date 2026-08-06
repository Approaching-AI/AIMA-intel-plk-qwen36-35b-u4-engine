# ADR 0033: Accept 1024 live state and select greedy tokens

Date: 2026-07-11

## Status

Accepted for one 1024-token sequential routed-MoE live-state row. Multi-case
teacher-forced distribution, deterministic continuation, context ladder, and
product speed remain open.

## Context

ADR 0032 accepted independent all-layer components but required their errors to
propagate through the real graph. Clean seq687 runs paired otherwise-identical
CPU-reference evaluations. In the candidate row, each layer's live
attention-post-norm, router IDs, and normalized weights feed the resident
native component; its routed output is compared to the same-state CPU result
and then overwrites `ffn_moe_out` before downstream graph execution.

All 40 ordered injections pass. The live same-state relative-L2 range is
`0.000208295..0.000501988`; the aggregate over `83,886,080` routed values is
cosine `0.999999955185`, relative L2 `0.000299383`, max abs `0.00153685`, with
zero values above `5e-3`.

The final full-vocabulary row compares all `248,320` logits. Baseline and
injected top-1 are both token `264`; KL divergence is `0.000369903` versus the
accepted `0.005` ceiling. One native context owns all `24,746,393,600` payload
bytes, and process links/maps exclude oneDNN/OpenVINO.

Logit relative L2 is `0.101566`, which is not itself the distribution contract;
the accepted softmax KLD and top-1 gates are the relevant downstream ruler.

Evidence:

- `output/all-layer-live-state-injection-20260711Tseq687cleanZ/`
- `output/all-layer-exact-q6-component-20260711Tseq686cleanZ/`

## Decision

Accept the 1024-token live-state routed-MoE injection row. Close first-step
sequential drift for this prompt and advance to deterministic continuation.

Extend the same paired harness to emit eight greedy token IDs after the 1024-
token prefill. The injected prefill state must persist into the continuation,
but the still-unported M=1 decode graph remains on the CPU reference; disable
the M=1024 injector during decode. Require exact equality of all eight IDs in
addition to the already-fixed 40 live component and first full-vocabulary KLD
gates.

This is a prefill-state correctness test, not proof of a native decode engine.
Its hybrid wall time remains ineligible for performance claims.

## Consequences

- Do not rerun isolated component or single-step live-state gates unless carrier
  math changes.
- A token pass advances to broader prompt/context rows; a miss localizes the
  first divergent continuation position.
- Native M=1 decode correctness and performance remain separate open work.

## Follow-Up

- Add guarded greedy continuation to the existing injection harness.
- Rerun one clean paired row with eight exact token IDs required.

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
