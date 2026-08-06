# ADR 0057: Close paged provider and select INT8-KV GQA

Date: 2026-07-13

## Status

Accepted. The fixed product paged-GQA provider source is rejected; one native
blockwise-INT8 KV component is selected. Product promotion remains open.

## Context

ADR 0056 registered one exact product-pipeline source/rate gate with a
`28.250 ms` budget across the ten full-attention layers. Clean seq777 captures
the actual long-context program family rather than the 2k fallback:

- nine layers use hash `12004046352395215748`, one uses
  `1508032723876328546`;
- each layer dispatches one `paged_attention_opt__gqa_single_token` main and
  one `single_token_finalization` kernel;
- all 60 dispatches map to three isolated, successfully disassembled binaries;
- token attention sums are `33.109`, `31.229`, and `27.952 ms`;
- registered repeat/confirm is `31.229 / 27.952 ms`, with `11.73%` spread.

The confirm row alone clears the cap, but the repeat row misses by `10.54%`
and the pair is far outside the `0.5%` noise band. The registered gate requires
both. A favorable third row cannot promote an unstable source, and ADR 0056
excluded cache/block/partition/property sweeps.

All native and provider routes measured so far store full-attention KV as F32
or FP16. KV precision is not locked by the model or target contracts; only
output correctness is locked. At 128k, FP16 K+V traffic is 256 MiB per layer
before metadata. A fixed per-token, per-head, 32-dimension symmetric INT8
representation reduces each tensor from 128 MiB to 64 MiB plus 4 MiB of FP16
scales. That changes the dominant bandwidth term rather than tuning the closed
FP16 work distribution.

## Decision

Close the product paged-GQA provider family. Select exactly one native
compressed-KV component:

- signed INT8 K and V;
- one FP16 symmetric max-absolute scale per 32 dimensions, per token and KV
  head; round-to-nearest-even and clamp to `[-127,127]`;
- fixed query heads `16`, KV heads `2`, head dimension `256`, context
  `131072`, chunk `256`, WG256/SIMD32, and eight-way GQA-local reuse;
- load each INT8 K/V element once per KV group/chunk, dequantize into local
  memory, then use the existing online-softmax and bounded partial reduction;
- include one-token K/V quantization, partial attention, and final reduction in
  the component time; perform zero timed host transfer.

The terminal gate is finite output, cosine `>=0.999`, relative L2 `<=0.002`,
both repeat and confirm complete time `<=2.825 ms/layer`, and paired spread
`<=0.5%`. A pass admits 32k/64k guards and packed-backend integration. A
numeric, rate, or noise failure closes INT8 KV plus scale datatype, group size,
rounding, chunk, local size, subgroup, and workgroup variants without a sweep.

## Consequences

- Do not replay or integrate seq777's paged provider binaries.
- The prior FP16 scalar and XMX work distributions remain closed. This route
  is admitted only because it halves the dominant storage/traffic term and
  adds a locked quantization accuracy boundary.
- A component pass is not token correctness, output-512 performance, or a
  speedup claim.

Evidence:

- `output/openvino-paged-gqa-provider-20260713Tseq777cleanZ/`
- `output/fused-gqa-fp16-kv-decode-20260713Tseq772cleanZ/`
