# Selected Gate Q4-Plane Pair Candidate

Workstream: `intel-qwen36-35b-a3b-gguf-q4km`

Status: R3 default-off candidate. This is not a promoted speedup claim.

## Route

Flag: `--selected-gate-q4-plane-pair-dot`

The route fuses selected-expert gate/up q4-plane row-pair dot work so the two
adjacent rows share one Q8 input traversal. The default route remains unchanged
unless the flag is explicitly enabled.

## Evidence

Primary rollup:

- `output/r3-selected-gate-q4-plane-pair-rollup-20260629T125530Z/`

Profile A/B:

- baseline: `output/r3-q4-plane-gap-rollup-20260629T093021Z/`
- trial: `output/r3-q4-plane-gap-rollup-20260629T092537Z/`
- total profile ratio: `0.9648x`
- selected gate/up op ratio: `0.9135x`

Timing A/B rows covered by the rollup:

- 001k: `sentinel_001k`, `prefill_shape_001k`
- 002k: `sentinel_002k`, `prefill_shape_002k`
- 004k: `sentinel_004k`, `prefill_shape_004k`

Rollup result:

- required checks passed
- all selected timing rows generated 512 tokens
- top-k signatures stayed stable
- case-total ratio range: `0.9670x..0.9912x`
- prefill ratio range: `0.9621x..0.9895x`
- decode tok/s ratio range: `1.0001x..1.0256x`

## Caveats

- `output/r2-native-matrix-20260629T104717Z/` is a partial artifact for the
  full 004k default-EOS bucket because `sentinel_004k` stopped at 495/512
  tokens. The `prefill_shape_004k` row in that artifact did generate 512/512
  tokens and is used only as a row-level A/B input.
- `sentinel_004k` fixed-token evidence uses `--ignore-eos`:
  `output/r2-native-matrix-20260629T113427Z/` versus
  `output/r2-native-matrix-20260629T115743Z/`.
- No valid 008k/016k timing evidence has been collected for this route.
  `output/r3-selected-gate-q4-plane-pair-008k-cost-stop-20260629T135123Z/`
  records an aborted 008k probe, but it missed
  `--selected-expert-down-q6-pair-dot` and is not timing evidence.
- Promotion-grade benchmark discipline is not satisfied.

## Decision

Keep the route as a default-off R3 candidate. Only rerun 008k/016k timing with
the exact packaged flags and explicit target-time reservation.

Do not enable by default or make speedup claims until correctness, smoothness,
and benchmark coverage are promoted through the project gates.
