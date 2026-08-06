# ADR 0054: Close scalar fused GQA and select XMX flash decode

Date: 2026-07-13

## Status

Accepted. The scalar fused component is rejected; one fixed XMX successor is
selected. Product promotion remains open.

## Context

ADR 0053 selected one 128k GQA-aware FP16-KV component with a hard
`2.825 ms/full-attention layer` repeat/confirm cap. Commit `29cd0a9` implements
the fixed chunk-256, SIMD32 scalar-FMA design. It reads each K/V tile once,
shares it across eight GQA heads, performs online softmax, emits bounded
partials, and reduces them without host transfer.

Clean seq772 is numerically strong: output cosine is `0.99999999946`, relative
L2 is `8.17e-5`, and maximum absolute error is `1.85e-6`. Timing is stable but
misses the registered cap:

- repeat: `2.591875 + 0.252500 = 2.844375 ms`;
- confirm: `2.594687 + 0.252708 = 2.847395 ms`;
- paired spread: `0.106%`;
- cap miss: `0.019-0.022 ms`, or `0.69-0.79%`.

The component cannot be accepted by rounding or a favorable dirty-tree row.
Chunk, subgroup, datatype, and workgroup variants were excluded in advance.

## Decision

Close the scalar/subgroup-FMA partial-plus-reduce shape. Select one materially
different matrix-engine flash-decode component:

- fixed token tile `16`, context chunk `256`, workgroup `256`, subgroup `16`;
- one workgroup per KV-head/context chunk;
- FP16 query and DPAS-ready transposed K tiles;
- one `8x16x256` XMX score product for all eight GQA heads and 16 tokens;
- online softmax weights shared in local memory;
- sixteen parallel `8x16x16` XMX value products covering all 256 output
  dimensions while reading V once per GQA group;
- the same bounded partial reduction and zero timed host transfer.

The hard gate remains cosine `>=0.999`, relative L2 `<=0.002`, both repeat and
confirm `<=2.825 ms/layer`, and spread `<=0.5%` at 128k. A miss closes the XMX
shape without a tile or subgroup sweep. A pass admits 32k/64k guards and
backend integration, not output512 or product speed.

## Consequences

- Do not integrate commit `29cd0a9`'s scalar component into the token backend.
- Preserve its numeric and bandwidth evidence; do not retry native-exp,
  chunk-size, local-size, subgroup, or FP16/BF16 variants.
- The XMX source must expose the fixed tile and layout in a source gate before
  timing.

Evidence:

- `output/fused-gqa-fp16-kv-decode-20260713Tseq772cleanZ/`
