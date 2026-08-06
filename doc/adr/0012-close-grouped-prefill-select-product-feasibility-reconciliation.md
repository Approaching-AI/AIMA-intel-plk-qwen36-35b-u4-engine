# ADR 0012: Close grouped prefill and select product feasibility reconciliation

Date: 2026-07-11

## Status

Accepted; supersedes ADR 0011 as the active route decision.

## Context

ADR 0011 authorized one static repository-owned source/ISA and complete-flow
gate before any further real kernel. Clean seq651 proves that the local
arithmetic is not the blocker:

- the PTL compiler emits three native `dpas.8x8` U4 instructions for paired
  gate/up and down epilogues;
- the generated path keeps F32 affine restoration ahead of the F16 store;
- rounding the exact post-affine SwiGLU oracle to F16 produces zero values
  above `5e-3` across all `4,194,304` values (max abs `0.0038414`).

The complete dataflow nevertheless fails. The grouped wrapper dispatches and
stores compact rows by expert. A token's eight selected-expert contributions
therefore belong to distinct workgroups, which have no grid barrier for a
deterministic in-kernel reduction. Atomics are nondeterministic; changing work
ownership back to token/M1 reopens the already closed handwritten topology;
eight rank-serial launches reread expert weights eight times. The relaxed F32
partial-plane projection is already `8561.916 us` versus the `8514.926 us`
cap while omitting SwiGLU exponential and synchronization costs.

Seq652 tests the only bounded carrier relaxation rather than assuming that
the partial plane must be F32. It rounds every normalized weighted-down value
to F16 and performs the deterministic eight-rank scatter in F32:

- weighted down: all `16,777,216` values, zero mismatches above `5e-3`, max
  abs `5.60264e-5`, RMSE `1.01846e-6`;
- routed output: all `2,097,152` values, zero mismatches above `5e-3`, max abs
  `5.62966e-5`, RMSE `2.88273e-6`;
- paired PTL codegen still emits three native U4 DPAS instructions;
- the measured pure-work probes execute `805,306,368` F32 residual FMAs in at
  most `58.854 us` and `4,194,304` SwiGLU values in at most `51.250 us`.

Correctness and scalar math therefore pass, but the real exact-group32/F32-
scale matrix core does not. Using seq650's complete minimum minus all four
external stages gives an optimistic `7944.995 us` exact-core basis. After
crediting paired gate/up input and output traffic, adding exact floating-min
payload delivery, measured math, gather, and the existing scatter, the
source-realizable projection is `9539.674 us`. Even eliminating gather and
byte-scaling scatter to the F16 carrier leaves `8718.387 us`, still
`203.461 us` above the cap. No unmodeled work is charged in that ideal row.

## Decision

Close these routes without a real kernel:

- `context_wide_1024_in_core_exact_q4k_affine_codegen_feasibility_v1`;
- `context_wide_1024_f16_deterministic_contribution_prefill_v1`.

Together with seq646, seq647, and seq650, this closes the context-wide grouped
exact-Q4_K prefill family: separated buffers, handwritten resident DPAS,
public grouped F16, in-core no-plane epilogues, and the sole compressed
deterministic partial carrier. Do not sweep generator strategy, datatype,
grouping, workgroups, APIs, GRFs, queues, atomics, rank-serial launches, or
zero-point approximations.

Select `locked_product_architecture_feasibility_reconciliation_v1`. Before any
more implementation, run exactly one product-level architecture audit:

1. inventory every closed prefill family and its measured reopen condition;
2. recompute the fastest evidence-backed exact 8k prefill bound against the
   `2781 tok/s` / `1.25x OpenVINO` contract;
3. inspect CPU, GPU, NPU, and legal hybrid execution for one materially
   independent architecture not already rejected, with a source-derived
   complete bound before implementation;
4. if no independent architecture clears the bound, record that the product
   target is infeasible under the simultaneous locked model, correctness,
   batch-1, and no-OpenVINO/oneDNN-runtime constraints, then request an owner
   decision rather than weakening any contract silently.

This audit may select one new architecture only if it is independent of the
closed grouped/handwritten/bucket families and clears the complete product
kill-number arithmetically. It may not lower the target or correctness bar.

## Consequences

- F16 contribution storage is retained as correctness evidence, not promoted
  as a performance route.
- No real seq651/652 matrix kernel exists or is authorized.
- Decode remains unresolved at the `52.79 tok/s` headline floor; improving
  decode alone cannot satisfy a product whose prefill path is infeasible.
- No native acceptance-matrix row or speedup is claimed.
- The project goal remains active pending the product feasibility audit.

## Evidence

- `output/in-core-affine-codegen-feasibility-gate-20260711Tseq651cleanZ/`
- `output/f16-contribution-plane-feasibility-gate-20260711Tseq652cleanZ/`
- `output/onednn-grouped-q4k-moe-component-gate-20260711Tseq650cleanZ/`
- `output/onednn-grouped-u4-moe-preflight-gate-20260711Tseq649cleanZ/`
- `doc/adr/0011-close-f16-grouped-residual-select-in-core-affine-codegen-feasibility.md`

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
