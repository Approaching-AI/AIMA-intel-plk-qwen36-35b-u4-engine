# ADR 0047: Close F16/U4 complete FFN and select product route reconciliation

Date: 2026-07-12

## Status

Accepted. No native prefill or product speed row is accepted here.

## Context

Commit `e53bf3f` implements the sole ADR-0046 source: exact pinned OpenVINO and
oneDNN offline codegen, GGUF-derived group-32 F16/U4 packages for all routed
and shared experts, and a native OpenCL-only runtime. Clean seq764 executes the
restored layer-27 `ffn_out` boundary with 222 routed experts plus the shared
expert. It maps no OpenVINO/oneDNN runtime library and performs zero timed host
upload or readback.

All three fused programs contain DPAS and compile spill-free at SIMD16, 256
GRFs, four EU threads, and one barrier. Repeat/confirm are stable: complete
device spans are `11537.292 / 11556.563 us`, with `0.167%` paired spread. The
matrix kernels alone take `7853.124 / 7865.728 us`, already
`1603.124 / 1615.728 us` above the complete `6250 us` cap before gather,
SwiGLU, residual, scalar-gate, and scatter work.

The final boundary also fails component accuracy. Both rows produce cosine
`0.9996892` but relative L2 `0.0249286` versus the locked `0.002` maximum.
Thus neither helper fusion nor framework/launch removal can rescue the timing
failure, and correctness cannot be promoted independently.

## Decision

Close `native_prefill_f16_u4_active_expert_microkernel_complete_ffn_source_v1`.
Do not vary datatype, scale representation, tile, subgroup, workgroup, expert
bucket, provider setting, or synthetic assignment shape.

Select exactly one evidence-only
`native_prefill_product_route_reconciliation_v1` gate. It must reconcile the
1024-token product cap with the fastest measured FFN matrix-only, linear-state,
and linear-projection rows; compare the result with the same-host OpenVINO
kernel profile; and inventory the already closed CPU, GPU, NPU, and hybrid
families. It may admit a successor only if a materially independent complete
architecture clears the product kill-number arithmetically. It must not run a
new target kernel or silently weaken model, correctness, batch, runtime, or
performance contracts.

## Consequences

- The source-exact codegen/runtime remains reproducible terminal evidence, not
  an engine path.
- F16/U4 accuracy failure and the matrix-only cap miss are independent stop
  reasons; repairing only one cannot reopen the source.
- Native prefill and the product matrix remain open. Route rejection does not
  complete the project goal.

Evidence:

- `output/complete-ffn-microkernel-source-gate-20260712Tseq764cleanZ/`
- `output/complete-ffn-boundary-gate-20260712Tseq763cleanZ/`
- `doc/adr/0046-close-routed-m8-restore-complete-ffn.md`

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
