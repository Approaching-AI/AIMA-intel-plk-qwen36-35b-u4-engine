# ADR 0035: Close Q6 F16 contribution and select F32

Date: 2026-07-11

## Status

Accepted as a codec-level attribution and one bounded precision repair. No
product correctness or speed promotion is implied.

## Context

ADR 0034 preregistered exactly two live-state attribution rows while still
executing and comparing every native component.

Clean seq689 overwrites only the 20 Q4-down layers. It passes all gates:

- aggregate same-state routed relative L2 `0.000259563`;
- first full-vocabulary KLD `0.000768802`, top-1 token `264` in both rows;
- exact continuation
  `[264,264,271,248068,198,8160,579,264]` in both rows.

Clean seq690 overwrites only the 20 exact-Q6-down layers. Components and the
first distribution still pass—aggregate relative L2 `0.000273421`, KLD
`0.000783157`, top-1 token `264`—but continuation diverges at zero-based
position 2. The Q6-only IDs are
`[264,264,198,59865,18,21,829,7320]`.

The mixed seq688 and Q6-only seq690 share the same first divergence position,
while Q4-only is token exact. This confines the next repair to the exact-Q6
carrier. That kernel accumulates each output in F32 but converts the weighted
contribution to F16 before the deterministic scatter. The all-layer R2 route
has not yet tested preserving that common boundary in F32.

Evidence:

- `output/all-layer-live-state-q4-attribution-20260711Tseq689cleanZ/`
- `output/all-layer-live-state-q6-attribution-20260711Tseq690cleanZ/`
- `output/all-layer-live-state-injection-tokens-20260711Tseq688cleanZ/`

## Decision

Accept Q4-only token-state correctness for this 1024-token row and reject the
current exact-Q6 F16-contribution state.

Add one exact-Q6 kernel variant that keeps its existing F32 accumulator,
router multiplication, contribution plane, and deterministic scatter in F32.
Do not change Q6 signed values, per-16 scales, source Q8 quantization, schedule,
tile, workgroup, layer set, or prompt. Q4 continues to use its accepted F16
carrier.

The fixed gate first reruns all-40 components, then the Q6-only and mixed
eight-token live-state rows. Every component must retain finite, cosine
`>=0.999`, relative L2 `<=0.002`; both token rows must retain KLD `<=0.005`,
first top-1 equality, and all eight baseline IDs.

## Consequences

- Individual Q6 layers and alternative quantizers remain closed.
- F32 contribution is an R2 correctness repair; its larger traffic is measured
  but not required to meet the product cap.
- If Q6-only still fails, the next attribution moves one boundary earlier to
  source-Q8 activation quantization. If Q6-only passes but mixed fails, evaluate
  additive interaction before changing Q4.

## Follow-Up

- Add a separate F32-output exact-Q6 kernel, buffer, and scatter path to the
  parameterized runtime.
- Rerun all-layer component and the fixed Q6-only/mixed token gates.

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
