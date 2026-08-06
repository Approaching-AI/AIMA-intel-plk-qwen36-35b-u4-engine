# ADR 0058: Close INT8 noise failure and select packed INT6 KV

Date: 2026-07-13

## Status

Accepted. The block32-INT8 component is rejected by its registered noise gate;
one fixed packed-INT6 successor is selected. Product promotion remains open.

## Context

ADR 0057 registered one exact 128k component with seven-sample repeat and
confirm distributions. Clean seq778 is the first native source to clear both
the numeric boundary and the absolute attention budget:

- cosine is `0.999999971822`, relative L2 is `0.000324519358`, and output is
  finite;
- repeat/confirm medians, including current-token K/V quantization, attention,
  and reduction, are `2.459061 / 2.471978 ms` versus the `2.825 ms` cap;
- the medians differ by `0.522537%` versus the pre-registered `0.5%` noise
  limit.

The rate margin is material, but ADR 0057 made numeric, rate, and noise jointly
terminal. A favorable rerun would change the decision rule after seeing the
result. The INT8 source therefore cannot be integrated or promoted.

The remaining dominant term is compressed KV traffic. INT8 values plus FP16
block32 scales occupy `136 MiB` for K+V at 128k. Packing signed six-bit values
reduces the same fixed representation to `104 MiB`: `48 MiB` values plus
`4 MiB` scales per tensor. This is a `23.53%` storage/traffic cut from seq778,
not a workgroup or measurement variant of the rejected INT8 source.

## Decision

Close INT8 K/V and all scale-datatype, group-size, rounding, chunk, local-size,
subgroup, workgroup, measurement-sampling, and favorable-rerun variants. Select
exactly one packed signed-INT6 component:

- signed range `[-31,31]`, round-to-nearest-even, six-bit two's-complement
  packing, and one FP16 symmetric max-absolute scale per 32 dimensions;
- token/head-contiguous 192-byte value rows, with four six-bit values packed
  into three bytes; no padded byte plane;
- fixed query heads `16`, KV heads `2`, head dimension `256`, context
  `131072`, chunk `256`, WG256/SIMD32, and eight-way GQA-local reuse;
- include current-token K/V quantize-and-pack, partial attention, and final
  reduction device events with zero timed host transfer;
- retain two seven-sample distributions and use each distribution median as
  the registered repeat/confirm row.

The terminal gate remains finite output, cosine `>=0.999`, relative L2
`<=0.002`, both medians `<=2.825 ms/layer`, and paired median spread `<=0.5%`.
Any failure closes packed INT6 plus packing order, scale representation, group,
rounding, chunk, local size, subgroup, and workgroup without a sweep. A pass
admits 32k/64k guards and packed-backend integration; it is not a product
speedup claim.

## Consequences

- Seq778 remains valuable rate and accuracy evidence but is not promoted.
- Do not rerun INT8, alter its sample count, or integrate it behind a flag.
- Native prefill and the full seven-context output-512 product matrix remain
  open regardless of the packed-INT6 component outcome.

Evidence:

- `output/compressed-gqa-i8-kv-decode-20260713Tseq778cleanZ/`
- `doc/adr/0057-close-paged-provider-select-int8-kv-gqa.md`
