# ADR 0008: Close handwritten DPAS and select oneDNN U4 Q4_K component

Date: 2026-07-11

## Status

Accepted; supersedes ADR 0007 as the active route decision.

## Context

Clean seq639 executes ADR 0007's sole 1024-token census. Across both locked
8k prompts, the normalized full-layer plus permutation floor is at most
`417.316 us/64 tokens` and averages `396.422 us` at `115 GB/s`, below the
`575.33 us` whole-layer budget. The worst-dispersion selected row is
`prefill_shape_008k`, layer 27: `222` active experts, `8192` assignments,
mean active `M=36.901`, and maximum `M=361`.

Charging the rest of that layer and the permutation stream first leaves a
hard `5479.754 us` window for gate/up through SwiGLU over all 1024 tokens.
Two exact handwritten OpenCL implementations miss it:

- seq640's M1/U8 DPAS kernel compares all `4,194,304` values at max absolute
  error `7.153e-6` and cosine `1`, but takes `48.687 ms` minimum
  (`6.331 GB/s`), `8.89x` over the cap;
- seq641 proves the PTL compiler emits native `dpas.8x8` with `:u4` source
  precision; seq642 uses that intrinsic and remains exact at max absolute
  error `6.199e-6` and cosine `1`, but takes `26.081 ms` (`13.989 GB/s`),
  `4.76x` over the cap. A large-GRF diagnostic removes spills but regresses.

The failure is the handwritten schedule, not absence of low-bit XMX support.
Clean seq643 pins oneDNN commit
`01b479323f794da1a7a41a6fc084c7e11ccc2c3b` and maps the same layer-27
histogram to seven power-of-two batched U4 JIT-GEMM calls. Every active
expert's `2048x1024` U4 weight tensor appears once. Padding raises `8192` real
assignments to `12352` (`50.781%`), yet the complete raw-U4 schedule takes
`1542.369 us` minimum and `1567.204 us` median—only `28.15%` of the cap.

This raw core intentionally omits Q4_K group scale, affine min compensation,
and SwiGLU. It proves implementation headroom, not component correctness or
product prefill speed.

## Decision

Close both handwritten expert-bucket DPAS routes and select
`context_wide_1024_onednn_u4_exact_q4k_prefill_v1`.

Authorize exactly one real layer-27 correctness-bearing component:

1. preserve seq639's expert assignments and seven-bucket schedule;
2. repack the real Q4_K gate/up codes to resident oneDNN U4 weights without
   changing a code;
3. apply the exact per-32-value Q4_K scale term and its separate affine min
   compensation against the component's Q8 inputs;
4. produce the same gate/up-to-SwiGLU boundary and compare all `4,194,304`
   values against the existing captured oracle;
5. time the oneDNN calls plus all runtime scale/min compensation and SwiGLU,
   excluding only one-time resident weight preparation, and require minimum
   time `<=5479.754 us`.

The component stops on either correctness or timing failure. No M bucket,
workgroup, API, register, scale precision, or compensation approximation sweep
is authorized. oneDNN is pinned for this gate; changing its commit is a new
route decision.

## Consequences

- M1/U8 and handwritten M8/U4 OpenCL scheduling are closed. Their exactness
  evidence remains useful, but neither can enter the target-facing schedule.
- The oneDNN raw-U4 result has `3937.385 us` of measured cap headroom for the
  missing exact operations; that headroom is the reason to implement one
  component, not a projected product speedup.
- Integer zero points alone cannot represent Q4_K's independent scale and min
  terms. The successor must account for the affine min term explicitly.
- Resident repacking may be outside the hot loop, but all per-prefill
  permutation, quantization, dequantization, compensation, activation, and
  scatter work remains chargeable by later whole-layer evidence.
- Decode remains unresolved at the `52.79 tok/s` headline floor, and no native
  acceptance-matrix row is promoted.
- The project goal remains active; route rejection or component admission is
  not completion.

## Evidence

- `output/prefill-router-shape-census-gate-20260711Tseq639cleanZ/`
- `output/expert-bucket-dpas-component-gate-20260711Tseq640cleanZ/`
- `output/m8-u4-dpas-preflight-gate-20260711Tseq641cleanZ/`
- `output/expert-bucket-dpas-component-gate-20260711Tseq642cleanZ/`
- `output/onednn-u4-bucket-preflight-gate-20260711Tseq643cleanZ/`

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
