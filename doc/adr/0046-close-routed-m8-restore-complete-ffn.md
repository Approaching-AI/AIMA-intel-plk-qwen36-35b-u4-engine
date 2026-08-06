# ADR 0046: Close routed M8 and restore the complete FFN boundary

Date: 2026-07-12

## Status

Accepted for one source-exact complete-FFN gate. No native prefill or product
speed row is accepted here.

## Context

Clean seq761 executes the sole ADR-0045 M8 source. Both kernels compile
spill-free at SIMD16, 96 GRFs, and ten EU threads, and all routed seq749
oracles pass. The measured matrix minima are `6442.333 / 6534.345 us`, only
`4.453 / 4.390 TMAC/s` versus the registered `5.4 TMAC/s` floor; paired spread
is `1.428%` versus `0.5%`. Gate/up alone takes `5052.749 / 5211.674 us`, so the
failure is row/weight reuse at N32, not router or dispatch overhead.

The source audit found a more fundamental boundary error. Seq749, seq759, and
seq761 stop at llama.cpp callback `ffn_moe_out`, which is only the routed
branch. Qwen3.6 then evaluates a shared expert, applies its scalar sigmoid
gate, adds it to the routed output, and exposes callback `ffn_out`. The model
contract already names `shared_expert` and `moe_residual`.

Layer 27 has routed gate/up `[2048,1024,256]`, routed down
`[512,2048,256]`, and three shared matrices `[2048,512]`, `[2048,512]`, and
`[512,2048]`. At 1024 tokens the shared branch adds `3.221B` MACs, or `12.5%`
over routed true work. Adding it to the fixed M8 padded schedule yields
`31.910B` matrix MACs. With the existing `626.566 us` non-matrix charge, the
corrected rate floor is `5.675 TMAC/s`; at the old `5.4` floor the complete
boundary is `6535.874 us`, already above the `6250 us` cap.

Clean seq762 then audits the exact installed OpenVINO commit
`90214e5be052438cec5617ed3ea7e37df1538f68`. Its default prefill route is
oneDNN grouped GEMM; its alternate active-expert micro route uses F16 gathered
activations, U4/I4 weights, F16 per-group scales, F32 accumulation, runtime
expert lengths/offsets, and optional SLM. On the target, warmed 1024-token
hidden-body medians are `542.531 / 525.676 / 554.514 ms` for default grouped,
micro, and per-expert loop. The micro architecture is `1.032x / 1.055x` faster
than those two controls. These synthetic PERF_COUNT rows select direction;
they are not the product denominator or a correctness claim.

## Decision

1. Close `fixed_m8_expert_major_routed_ffn_source_v1`. Do not vary its tile,
   N width, subgroup, workgroup, codec, or expert bucket.
2. Correct the promotion boundary from `ffn_moe_out` to `ffn_out`. Every future
   complete-FFN gate must include the shared matrices, scalar gate, routed plus
   shared add, and compare the final 1024x2048 output.
3. Admit exactly one source-exact F16-by-U4 active-expert microkernel gate based
   on the pinned OpenVINO generator/ABI. OpenVINO and oneDNN may be offline
   build/codegen dependencies only; the measured executable must map neither.
4. The gate must use real layer-27 assignments and full boundary captures,
   perform zero timed host upload/readback, pass cosine `>=0.999`, relative L2
   `<=0.002`, and finite checks at `ffn_out`, and clear `6250 us` in repeat and
   confirm within `0.5%`.
5. Any codegen/build, compiler-resource, full-boundary correctness, cap, or
   noise failure closes this source. It does not authorize datatype, tile,
   subgroup, workgroup, expert-bucket, or synthetic-assignment sweeps.

## Consequences

- Seq749 remains a correct routed-control reference, not a complete FFN.
- Generic materialized-F16 grouped paths remain closed. The only bounded
  reopening is the source-exact active-expert micro architecture selected by
  seq762's target differential.
- A source-gate pass would still be a component result. Native product prefill,
  long-context sentinel/smoothness, and the full acceptance matrix remain open.

Evidence:

- `output/fused-expert-ffn-m8-source-gate-20260712Tseq761cleanZ/`
- `output/complete-ffn-route-reflection-gate-20260712Tseq762cleanZ/`
- `output/openvino-hidden-prefill-profile-20260712Tseq751cleanZ/`
- `contracts/qwen36-35b-a3b-gguf-q4km-model-contract.json`

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
