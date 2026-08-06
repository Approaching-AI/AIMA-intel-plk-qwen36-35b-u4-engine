# ADR 0063: Integrate INT8 KV with an F32 hot window

Date: 2026-07-13

## Status

Accepted for packed-backend integration gates. This is not product promotion.

## Context

Seq783 admitted the exact signed-INT8 block32 component to packed-backend
integration. Direct integration preserved all teacher-forced top-1 decisions
and the generated token sequence, but accumulated error failed the full-vocab
distribution gate: maximum KLD was `0.0102346742687` versus `0.005`.
Component accuracy therefore did not transfer unchanged through forty layers.

A recent-token F32 tier isolates the error that compounds most strongly while
retaining INT8 plus FP16 block scales for older context. A `2048`-token F32
window passed the same distribution ladder at maximum KLD
`0.00389465741953`, but its 4k twenty-token diagnostic median was
`21.024320 ms`, above the `20.648358 ms` absolute cap. The smallest failing
core guard is 4k, so the integration window is fixed at `4096` tokens rather
than opening a window-size sweep.

Two lower-traffic recovery attempts were rejected before promotion:

- FP16 K and V in the hot tier failed at maximum KLD `0.0155209024468`;
- F32 K plus FP16 V failed at maximum KLD `0.00977000757865`.

These failures show that both hot K and hot V require F32 under the current
product correctness ruler.

## Decision

The packed Level Zero backend gets one default-off integration route with:

- signed INT8 K/V and one FP16 symmetric scale per 32 values for older tokens;
- a fixed `4096`-token circular F32 K/V hot tier;
- chunk256, WG256/SIMD32 partial attention and eight-query-head GQA reuse;
- separate compressed and hot state upload/readback semantics;
- the existing teacher-forced distribution gate plus twenty-sample one-sided
  95% latency upper bounds at exact core contexts.

The implementation also splits compressed and hot token loops so the hot path
does not carry a per-token codec branch. It does not authorize a codec, hot
window, chunk, subgroup, or workgroup sweep.

## Consequences

- Direct all-INT8 packed integration is closed by product-distribution error,
  even though its component row passed.
- FP16 hot-tier variants are closed by product-distribution error.
- The F32-hot4096 route must pass a clean short correctness bundle and exact
  `2k/4k/8k/16k/32k/64k/128k` twenty-sample guard before any output-512 work.
- Zero-initialized exact-context rows remain cost diagnostics. They cannot
  claim semantic correctness or an OpenVINO speedup.

## Follow-Up

- Commit the integration source, then run the clean correctness and exact-core
  guard tools from that commit.
- If any guard confidence interval misses its cap, profile or change route;
  do not select a favorable median or lower the product target.

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
