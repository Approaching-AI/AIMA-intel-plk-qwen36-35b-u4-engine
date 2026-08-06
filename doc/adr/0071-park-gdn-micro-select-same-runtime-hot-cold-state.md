# ADR 0071: Park GDN micro-tuning and select same-runtime hot/cold state

Date: 2026-07-14

## Status

Accepted

## Context

OV1 established a bit-exact all-layer GatedDeltaNet custom-operation carrier,
but clean seq805 remained `2.846x` slower than stock. Clean seq806 captured the
actual programs and identified dynamic pitch-array indexing, not spill, as the
cause. Clean seq807 removed that defect: indirect-stateless references fell
from `75` to zero, GRFs from `160` to `96`, EU threads rose from six to ten,
and all-30 custom GDN fell from `112.368 ms` to `49.170 ms`. Component outputs,
final logits, and all 80 states remained bit-exact. Same-run stock was still
`39.287 ms`.

The next two bounded source candidates failed to move the component gate.
Suppressing the compiler's time-loop unroll reached `51.449 ms`; explicit
`half4` output stores reached `50.858 ms`. The roadmap's two-candidate switch
rule therefore applies.

The highest-priority parked route was same-runtime long-context state. The
locked model embeds F16 KV state. Clean seq810 tested the direct
`KV_CACHE_PRECISION=u8` property with one stock and two isolated U8 product
workers at exact 2k/output20. Both U8 workers were deterministic with each
other but differed from stock's greedy output, so the route fails correctness
before long-context timing.

Evidence:

- `output/openvino-gdn-codegen-20260714Tseq806-cleanZ/`
- `output/openvino-gdn-codegen-20260714Tseq807-scalar-index-cleanZ/`
- `output/openvino-gdn-codegen-20260714Tseq808-no-time-unroll-exploreZ/`
- `output/openvino-gdn-codegen-20260714Tseq809-vector-store-exploreZ/`
- `output/openvino-kv-precision-20260714Tseq810-cleanZ/`

## Decision

Keep seq807 as the accepted GDN numeric/compiler carrier and park standalone
GDN kernel micro-tuning. OV1 may resume only through an adjacent
projection/layout/GDN fusion with a source-derived complete bound under the
matching paired end-to-end protocol.

Make OV2 same-runtime hot/cold state the active route. Reject the simple U8
property. The admitted design is one parameterized full-attention custom state
carrier that:

1. constructs prompt-conditioned state from zero inside the same OpenVINO
   `InferRequest`;
2. keeps the most recent 8192 K/V tokens in F32;
3. stores older K/V as signed INT8 with one FP16 symmetric scale per 32 values;
4. fuses old-state dequantization into the accepted GQA arithmetic; and
5. leaves linear-attention recurrent and convolution state on the untouched
   stock OpenVINO path.

This decision does not accept native/GGUF state import, claim that existing
zero-state rows have product semantics, claim a product speedup, or close the
remaining prefill obligation.

## Consequences

- Do not vary GDN unroll, stores, vector widths, workgroups, subgroups, or
  launch parameters.
- Do not sweep `KV_CACHE_PRECISION` properties or spend priority long rows on
  the incorrect U8 route.
- One real full-attention layer must pass component, distribution, exact
  greedy-token, and persistent-state ownership gates before all-ten-layer or
  `32k/64k/128k` timing.
- State must not cross runtimes or requests. If the custom-operation ABI forces
  materialization, record that bound and use a derived exact-bucket OpenVINO
  state graph or close the route.

## Follow-Up

- Prove the one-layer packed attention/state graph ABI.
- Reuse the accepted block32-INT8 GQA arithmetic only after the stock OpenVINO
  boundary is locked.
- Supersede this ADR if the one-layer ABI/semantic gate closes or a different
  same-runtime state architecture has a complete correctness and rate bound.

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
