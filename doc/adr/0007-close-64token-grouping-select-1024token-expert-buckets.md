# ADR 0007: Close 64-token grouping and select 1024-token expert buckets

Date: 2026-07-11

## Status

Accepted; supersedes ADR 0006 as the active route decision.

## Context

Clean seq638 captures exact top-8 router assignments for the first real
64-token tile of both locked 8k `prefill_shape` and `sentinel` prompts. All 80
case/layer tensors have 512 valid assignments with prompt, layer, token, and
expert identity preserved.

The old DPAS prototype used the same eight experts for all 64 synthetic token
inputs, giving the theoretical maximum group size `M=64`. Real routing is much
more dispersed:

- `prefill_shape_008k`: mean/median/max active experts per layer are
  `95.425 / 93 / 155`; mean active group size is `5.461`;
- `sentinel_008k`: mean/median/max are `79.575 / 76.5 / 130`; mean active group
  size is `6.579`.

At the measured `115 GB/s` planning line, unique gate/up weights alone have an
average lower bound of `978.851 us/layer` and `816.265 us/layer` for the two
cases. Their combined mean is `897.558 us/layer`, before selected down, shared
expert, attention, router, activation, or arithmetic work. The entire 8k
layer budget is `575.33 us`. A 64-token grouped kernel therefore cannot pass
even with zero compute cost.

The tensor inventory identifies one non-swept successor. Stored per-layer
weights span `472,213,504..549,757,696` bytes. If expert work is bucketed over
a wider token window and each layer's weights stream once, the source-weight
memory floor normalized to 64 tokens is:

- 512-token window: `513.276..597.563 us/layer`—the worst layer already fails
  before compute;
- 1024-token window: `256.638..298.781 us/layer`—the first power-of-two window
  whose worst layer leaves positive whole-layer headroom.

## Decision

Close `target_facing_grouped_dpas_prefill_v2` at 64 tokens without a GPU
kernel. Pop the highest-ranked offline-repack/streaming-layout family and
select `context_wide_1024_expert_bucketed_prefill_v1`.

Its first unit is exactly one 1024-token real-router census on the same two
locked 8k prompts:

1. process tokens 0..1023 with correct recurrent and attention state;
2. preserve every top-8 assignment for all 40 layers;
3. build expert-major bucket shapes and report active experts, group-M
   distributions, source-weight bytes, activation-index bytes, and working
   set;
4. normalize the full stored layer-weight memory floor to 64 tokens and
   require every layer to remain below `575.33 us` at `115 GB/s`.

This is still a census/roofline gate. It may authorize one real expert-bucket
MxN DPAS component only if the exact 1024-token shapes pass the memory floor.
There is no 128/256/512 tile sweep; 1024 is selected by the locked full-layer
byte inventory above.

## Consequences

- The synthetic fixed-eight-expert prototype is an optimistic reuse case, not
  evidence for real router shapes.
- No 64-token grouped kernel is admissible under the product budget.
- Context-wide bucketing requires an expert-major activation permutation and
  inverse scatter; those bytes and storage must be charged by the 1024-token
  gate before kernel work.
- The gate must not claim prefill speed. A later component still needs DPAS
  ISA evidence, exact numeric comparison, repeat timing, and whole-layer
  integration.
- Decode remains unresolved at the `52.79 tok/s` product floor.
- The project goal remains active; route rejection is not completion.

## Evidence

- `output/prefill-router-shape-census-gate-20260711Tseq638cleanZ/`
- `output/dpas-storage-workdist-gate-20260707Tseq79Z/`
- `output/r1-native-gguf-load-map-20260705T071855Z/tensor-index.jsonl`

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
