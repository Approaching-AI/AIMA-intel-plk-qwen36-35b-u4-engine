# Dense Q4-Plane Pair Candidate

Workstream: `intel-qwen36-35b-a3b-gguf-q4km`

Status: R3 default-off candidate. This is not a promoted speedup claim.

## Route

Flag: `--dense-q4-plane-pair-dot`

The route reuses the existing q4-plane row-pair helper for generic dense Q4_K
matvec rows. It targets the largest remaining profile op,
`matvec_tensor_dense_q4plane`, and records the trial op as
`matvec_tensor_dense_q4plane_pair`.

The route requires `--q4-plane-layout` and remains default-off.

## Evidence

Profile A/B:

- rollup: `output/r3-dense-q4-plane-pair-profile-ab-20260629T141817Z/`
- baseline: `output/context-ladder-native-diagnostic-20260629T092603Z/`
- trial: `output/context-ladder-native-diagnostic-20260629T140904Z/`
- route check: `output/r3-q4-plane-route-check-20260629T141641Z/`
- prompt prefill ratio: `0.9829x`
- total profile ratio: `0.9790x`
- dense q4-plane op ratio: `0.9254x`
- top-k stable: true

001k timing A/B:

- rollup: `output/r3-dense-q4-plane-pair-001k-timing-ab-20260629T143151Z/`
- baseline: `output/r2-native-matrix-20260629T094503Z/`
- trial: `output/r2-native-matrix-20260629T141853Z/`
- required checks passed
- both rows generated 512/512 tokens
- top-k signatures stayed stable
- case-total ratio range: `0.9714x..0.9795x`
- prefill ratio range: `0.9739x..0.9833x`
- decode tok/s ratio range: `1.0274x..1.0338x`

002k timing A/B:

- rollup: `output/r3-dense-q4-plane-pair-002k-timing-ab-20260629T145744Z/`
- baseline: `output/r2-native-matrix-20260629T102306Z/`
- trial: `output/r2-native-matrix-20260629T143545Z/`
- required checks passed
- both rows generated 512/512 tokens
- top-k signatures stayed stable
- case-total ratio range: `0.9725x..0.9877x`
- prefill ratio range: `0.9673x..0.9879x`
- decode tok/s ratio range: `1.0126x..1.0131x`

004k timing A/B:

- rollup: `output/r3-dense-q4-plane-pair-004k-timing-ab-20260629T160842Z/`
- sentinel baseline: `output/r2-native-matrix-20260629T115743Z/`
- sentinel trial: `output/r2-native-matrix-20260629T152828Z/`
- prefill baseline: `output/r2-native-matrix-20260629T122518Z/`
- prefill trial: `output/r2-native-matrix-20260629T150212Z/`
- required checks passed
- both rows generated 512/512 tokens
- top-k signatures stayed stable
- case-total ratio range: `0.9803x..0.9880x`
- prefill ratio range: `0.9778x..0.9870x`
- decode tok/s ratio range: `1.0072x..1.0079x`

Candidate rollup:

- `output/r3-dense-q4-plane-pair-candidate-rollup-20260629T160842Z/`
- observed buckets: 001k, 002k, 004k
- case-total ratio range: `0.9714x..0.9880x`
- prefill ratio range: `0.9673x..0.9879x`
- decode tok/s ratio range: `1.0072x..1.0338x`

## Caveats

- Evidence covers 001k/002k/004k only. Do not infer 008k+, 016k+, or
  context-ladder smoothness.
- The CPU q4-plane pair family was closed by the 2026-06-29 GPU backend
  decision after two sub-threshold candidates. This residual 004k closure
  updates the record but does not reopen CPU micro-tuning.
- The route has not been combined with the selected gate/up q4-plane pair route.
- Promotion-grade benchmark discipline is not satisfied.
- `speedup_claims_allowed=false`.

## Decision

Keep the route default-off and closed under the GPU pivot. The 001k/002k/004k
evidence is positive and top-k stable, but it remains sub-threshold CPU route
evidence and must not be enabled by default or promoted as a speedup claim.
