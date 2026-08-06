# ADR 0003: Surrogate-refine split-plane dual-phase engine

Date: 2026-07-11

## Status

Superseded by ADR 0004 after the compact-Q6 killer gate failed.

## Context

The minimum product target is `1.25x` same-host OpenVINO GPU U4 in both
prefill and decode at every accepted context bucket. At 8k this is `2781`
prefill tok/s and `52.79` decode tok/s, or `18.943 ms/token` for decode.

The strict decode inventory is `1,975,676,544` active bytes/token before KV
traffic. Real full-tensor Q4 carriers reach `108.793-110.522 GB/s`, but the
original output head alone is `417,177,600` Q6_K bytes and all F32 routers are
`83,886,080` bytes. A representative F32 router dispatch reaches only
`5.639 GB/s`; the 40 sequential router dispatches therefore cannot fit the
new target unchanged. The real full-tensor raw-Q6 carrier reaches only
`52.720 GB/s`.

Two clean-tree component gates establish new source boundaries:

- `output/lm-head-q4-surrogate-gate-20260711Tseq627cleanZ/`: a derived Q4_K
  output-head surrogate recalls the exact-Q6 winner within rank 2 on all 96
  final-norm vectors. Recomputing the surrogate top 16 with the original Q6
  rows gives recall `1.0`, maximum hybrid KLD `0.001719558251`, and reduces
  head traffic from `417,177,600` to `286,091,520` bytes/token.
- `output/router-i8-surrogate-gate-20260711Tseq628cleanZ/`: row-scale I8
  router weights with block-32 I8 inputs place every exact top-8 expert within
  surrogate rank 10 on 77 distinct real inputs spanning all 40 layers.
  Recomputing the top 16 original F32 rows preserves all 77 exact expert sets
  and normalized weights, and reduces router traffic from `83,886,080` to
  `26,255,360` bytes/token.

After both cuts, active candidate traffic is `1,786,959,744` bytes/token and
the worst uniform-carrier demand is `105.076 GB/s`. A conservative
mixed-carrier budget leaves three coupled kill-numbers:

- compact Q6 carrier: `>=96 GB/s` on a real full tensor;
- fused router surrogate plus exact refinement: `<=0.450 ms/token` over all
  40 layers;
- all remaining queue/schedule overhead: `<=0.350 ms/token`.

At those bounds the worst short-context Q6 requirement is
`94.192 GB/s` at 8k. These are joint requirements, not independent speed
claims.

## Decision

Select `surrogate_refine_splitplane_dual_phase_v1` as the replacement
architecture.

Decode uses one offline-derived, device-resident layout owned by this runtime:

1. Q4_K tensors use the already proven x8 packed carrier.
2. The output head uses the Q4_K surrogate followed by exact-Q6 refinement of
   16 candidate rows.
3. Every router uses the row-I8/block-32-I8 surrogate followed by exact-F32
   refinement of 16 candidate rows. Expert selection and normalized weights
   are taken only from the exact refined values.
4. Remaining Q6_K work must use an exact compact low4/high2 split-plane DPAS
   carrier. It may repack offline, but it may not expand the persistent Q6
   payload into ordinary I8 rows or exceed the recorded resident-byte budget.
5. The token loop, weights, state, candidate selection, and refinement remain
   device-resident on a persistent queue; host bridges are outside the route.

Prefill is a separate token-blocked XMX/DPAS GEMM schedule with chunked
attention. At the 8k target, a 64-token tile has `575.33 us/layer` for the
whole parameterized layer. The prior token-reuse qmatvec prototype is not this
schedule and remains rejected.

This decision does not reopen or promote the seq626 rowblock16 26-mask route.
It does not authorize a native speedup claim. Head and router results are
component correctness and traffic evidence only; the replacement engine must
return to whole-model teacher-forced distribution before token, context, or
acceptance-matrix promotion.

## Consequences

- Pure surrogate head logits are forbidden: their observed maximum KLD is
  `0.03072181057`; exact candidate refinement is mandatory.
- Pure surrogate router selection is forbidden. Only the exact-refined expert
  IDs and weights may feed the MoE.
- Ordinary Q6-to-I8 row expansion, API swaps, local-size sweeps, and the old
  token-reuse prefill shape cannot satisfy this ADR.
- The compact-Q6 branch stops after one real full-tensor gate if it is below
  `96 GB/s` or fails exact Q6/Q8 component comparison. The architecture must
  then be revised rather than swept.
- The router GPU branch stops if the fused 40-layer projection is above
  `0.450 ms/token`; correctness alone is not a route signal.
- A prefill component must be a genuine multi-token GEMM and fit the
  `575.33 us/layer` 64-token whole-layer budget before expansion.

## Follow-Up

- Implement and measure the exact compact Q6 low4/high2 split-plane carrier on
  one real full tensor.
- Implement the fused I8-router/top16-exact-F32 GPU kernel and measure the
  whole-token 40-layer sum.
- Build one parameterized 64-token prefill layer using matrix-shaped work and
  chunked attention.
- Integrate only after the component kill-numbers pass, then run the locked
  teacher-forced ladder before any token or product benchmark.

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
