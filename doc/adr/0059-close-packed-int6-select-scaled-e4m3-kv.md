# ADR 0059: Close packed INT6 and select scaled E4M3 KV

Date: 2026-07-13

## Status

Accepted. Packed INT6 is rejected by its registered rate gate; one fixed
byte-aligned, block-scaled E4M3 successor is selected. Product promotion is
still open.

## Context

Clean seq779 measures ADR 0058's exact packed representation:

- cosine is `0.999999876270`, relative L2 is `0.000612114164`, and output is
  finite;
- repeat/confirm seven-sample medians are `3.134270 / 3.132083 ms`, including
  current-token quantize-and-pack, attention, and reduction;
- paired spread is a passing `0.069777%`, but both medians exceed the fixed
  `2.825 ms` cap by about `10.9%`.

The route reduces K+V storage from `136 MiB` to `104 MiB`, yet partial
attention regresses from seq778's `2.20 ms` range to `2.88 ms`. Cross-byte
six-bit extraction and sign reconstruction cost more than the saved traffic.
The registered route forbids packing/order/work-distribution variants, so the
source is terminal without integration.

Seq778 separately proves that a byte-aligned one-byte value plus FP16 block32
scale representation has sufficient absolute-rate margin: its slower complete
median was `2.471978 ms`, leaving `0.353022 ms` to the cap. A scaled E4M3
representation preserves that `136 MiB` traffic shape while replacing
cross-byte integer extraction with one byte load and one fixed 256-entry
E4M3-to-FP16 lookup. It is a distinct minifloat codec, not an INT8 rerun.

## Decision

Close packed INT6 and all bit order, packing width, scale representation,
group, rounding, chunk, local-size, subgroup, workgroup, and integration
variants. Select exactly one scaled E4M3 component:

- standard finite E4M3 encoding with bias 7, round-to-nearest-even, and one
  FP16 symmetric max-absolute scale per 32 dimensions, per token and KV head;
- byte-aligned K/V values and a compile-time 256-entry FP16 decode table;
- fixed query heads `16`, KV heads `2`, head dimension `256`, context
  `131072`, chunk `256`, WG256/SIMD32, and eight-way GQA-local reuse;
- include current-token K/V scale-and-encode, partial attention, and final
  reduction events with zero timed host transfer;
- retain two seven-sample distributions and register their medians.

The terminal gate remains finite output, cosine `>=0.999`, relative L2
`<=0.002`, both medians `<=2.825 ms/layer`, and paired spread `<=0.5%`. Any
failure closes E4M3 plus lookup representation, scale/group/rounding, chunk,
local size, subgroup, and workgroup without a sweep. A pass admits 32k/64k
guards and packed-backend integration but is not a product speedup claim.

## Consequences

- Do not optimize, repack, or integrate seq779's INT6 source.
- E5M2, alternate FP8 formats, table/storage encodings, and unscaled FP8 are
  not fallback variants; this decision registers E4M3 only.
- Native prefill and the seven-context output-512 matrix remain open.

Evidence:

- `output/packed-gqa-i6-kv-decode-20260713Tseq779cleanZ/`
- `output/compressed-gqa-i8-kv-decode-20260713Tseq778cleanZ/`
