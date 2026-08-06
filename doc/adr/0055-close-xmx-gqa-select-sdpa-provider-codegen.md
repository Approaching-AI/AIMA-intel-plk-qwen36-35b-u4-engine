# ADR 0055: Close fixed XMX GQA and select SDPA provider codegen

Date: 2026-07-13

## Status

Accepted. The fixed native XMX component is rejected; one exact-shape
OpenVINO SDPA offline-codegen successor is selected. Product promotion remains
open.

## Context

ADR 0054 admitted exactly one matrix-engine successor to the scalar fused-GQA
near miss. Commit `1ec3ad4` implements the registered 128k, token-tile-16,
chunk-256, WG256/SIMD16 shape. Clean seq773 satisfies the fixed source contract
and remains numerically strong:

- cosine `0.999999999413`;
- relative L2 `9.065e-5`;
- maximum absolute error `2.209e-6`;
- finite output.

It fails the terminal rate and noise gates:

- repeat: `5.863125 + 0.251145 = 6.114270 ms`;
- confirm: `5.897395 + 0.251666 = 6.149061 ms`;
- paired spread: `0.566%`, versus `0.5%` allowed;
- component cap: `2.825 ms`, so the faster row is `2.164x` over budget.

The result is not a near miss and does not authorize a tile, subgroup, chunk,
datatype, GRF, or instruction-mapping sweep. A materially independent source
is required.

OpenVINO's clean hidden-prefill profile already proves that the pinned GPU
provider can select an optimized FP16 SDPA kernel family named
`ocl::sdpa::opt__f16`. That row is mechanism evidence only: it is prefill at a
different shape and supplies no 128k decode rate. The product denominator at
128k nevertheless establishes that a complete provider token runs close enough
to the target to justify one exact-shape provider capture before abandoning
this hardware route.

## Decision

Close the fixed native XMX score/value implementation. Select one pinned
OpenVINO GPU SDPA provider offline-codegen gate with these phases and no option
sweep:

1. Build and profile exactly query length `1`, context `131072`, query heads
   `16`, KV heads `2`, head dimension `256`, FP16 KV, causal decode. The
   selected node must report `ocl::sdpa::opt__f16`; a fallback implementation
   closes the source.
2. Export or reproducibly capture the exact compiled GPU program and immutable
   dispatch metadata. This is an offline build step only. The timed component
   runner must map no OpenVINO or oneDNN runtime library and must perform zero
   timed host transfer.
3. Compare its output with the existing F32 component oracle at 128k. Require
   finite output, cosine `>=0.999`, and relative L2 `<=0.002`.
4. Require both repeat and confirm `<=2.825 ms/full-attention layer` and paired
   spread `<=0.5%`.

A capture failure, provider fallback, native replay failure, numeric failure,
or timing failure closes this provider source without a property, datatype,
layout, kernel-cache, tile, subgroup, or workgroup sweep. A pass admits
32k/64k guard rows and packed-backend integration; it is not output-512 or
product-speed evidence.

## Consequences

- Do not integrate the seq773 XMX kernel into the packed token backend.
- Keep the provider/compiler pinned and record the compiled-program hash,
  source shape, build properties, native runtime maps, transfer accounting,
  command, and raw timing output.
- The existing 1024-token hidden-prefill SDPA profile is not an acceptance row;
  exact 128k query-one evidence is mandatory.
- Final runtime dependency policy is unchanged: OpenVINO may generate the
  program offline, but the promoted runtime must remain native.

Evidence:

- `output/xmx-gqa-fp16-kv-decode-20260713Tseq773cleanZ/`
- `output/openvino-hidden-prefill-profile-20260712Tseq751cleanZ/`
