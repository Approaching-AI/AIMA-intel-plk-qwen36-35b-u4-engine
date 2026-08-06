# ADR 0074: Accept all-ten OpenVINO hot/cold attention semantics

Date: 2026-07-14

## Status

Accepted

## Context

ADR 0073 accepted one integrated layer-3 full-attention replacement. Expanding
that carrier to all ten full-attention layers exposed two composition effects:

1. Downstream candidate K/V can no longer be required to equal stock K/V once
   an upstream attention layer is custom. The valid state invariant is the
   candidate's own phase-to-phase preservation, append, eviction, and codec
   transition, while final distributions and greedy tokens remain compared
   directly with isolated stock OpenVINO.
2. Repeating the earlier scalar two-pass softmax across ten layers accumulated
   enough prefill-state drift to fail the first 8k decode row. Capturing the
   actual stock `sdpa_micro` programs showed a chunked running-maximum
   reduction, `native_exp2`, F16-rounded numerator weights for the V product,
   F32 denominator weights, and native reciprocal. Reproducing that order
   reduced the full ten-layer distribution error below the existing gate.

Token zero is also a persistent attention sink. Keeping it only in the older
I8 history was avoidable error, so the physical hot carrier now pins that token
exactly in slot zero. The recent-token ring retains its logical 8192-token
window plus one race-avoidance guard row. Each K/V hot state is therefore
`[1,2,8194,256]`: one exact sink row and an 8193-row recent/guard ring.

Clean seq836 at commit `9545fcf` replaces layers
`[3,7,11,15,19,23,27,31,35,39]` from one source. It executes ten custom
operations and zero stock SDPA operations, owns 60 custom full-attention states
alongside the 60 untouched linear-attention states, and passes all required
checks at exact 2k and across 8192-to-8194. The five stock-referenced KLD rows
are:

- 2k: `0.000467901926`, `0.000086405903`;
- 8k boundary: `0.001075534567`, `0.003292638160`,
  `0.000002323400`.

Stock and candidate greedy paths are exactly `[271,248068]` and
`[271,248068,198]`. Every layer advances cold length as `0 -> 1 -> 2`; hot
preservation, sink preservation, signed block32-I8 payloads, and F16 scale
bytes are exact under candidate-owned transition checks.

Evidence:

- `output/openvino-hot-cold-attention-20260714Tseq836-allten-cleanZ/`
- `output/openvino-hot-cold-attention-20260714Tseq835-allten-stocklike-exploreZ/`
- `output/openvino-hot-cold-attention-20260714Tseq823-allten-exploreZ/`

## Decision

Accept seq836 as the all-ten semantic carrier and close OV2 gate 2c.

1. Keep one parameterized graph/source across all ten full-attention layers.
2. Keep the exact sink token, logical recent hot8192 window, physical guard,
   append-only signed block32-I8 older state, exact F16 scale bytes, and O(1)
   graph-owned length metadata.
3. Preserve the captured stock reduction and rounding order unless a proposed
   replacement passes the complete all-ten distribution gate again.
4. For all-ten gates, validate candidate-owned state transitions rather than
   comparing downstream K/V bits with a stock graph that has different
   upstream activations. Continue comparing final distributions, greedy paths,
   untouched state schemas, and long-context sentinels directly with stock.
5. Do not admit the current scalar kernel to output512 rate testing. The next
   gate is a stock-shaped tiled prefill/decode carrier that preserves these
   semantics and proves a complete performance bound.

## Consequences

- All-ten semantic composition and same-request state ownership are no longer
  the blocker.
- Seq836 timing is diagnostic and is decisively non-promotable. Its attributed
  ten-layer attention envelope is `2564.787 ms` versus stock `13.126 ms` at 2k
  and `30003.860 ms` versus stock `161.844 ms` at 8k. The current one-query,
  256-work-item scalar shape therefore needs a structural rewrite, not local
  workgroup or arithmetic-option tuning.
- The captured stock geometry supplies that rewrite direction: prefill uses
  subgroup 16, local `16x8`, a 32-query tile, and a 128-key chunk; decode uses
  subgroup 16, local `16x16`, a 16-query tile, and a 256-key chunk. The next
  carrier must compute each score tile once, retain it for the V product, and
  preserve hot/cold state side effects without returning full history.
- Priority `32k/64k/128k` output512 timing remains blocked until the tiled
  carrier clears its component/profile admission gate. Product completion
  remains all seven buckets, both phases, and the paired confidence contract.

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
