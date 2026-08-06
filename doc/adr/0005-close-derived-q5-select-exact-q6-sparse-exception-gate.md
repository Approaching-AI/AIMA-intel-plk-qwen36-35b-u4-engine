# ADR 0005: Close derived Q5 and select an exact Q6 sparse-exception gate

Date: 2026-07-11

## Status

Superseded by ADR 0006 after the static source model fell below the
preregistered kill-number.

## Context

The product contract remains at least `1.25x` same-host OpenVINO GPU U4 in
both phases at every bucket. At the headline 8k lane, decode must reach
`52.79 tok/s`, leaving `18.94298 ms/token`; the non-head exact-Q6 lane after
the accepted head/router cuts has `3.74439 ms/token`.

Clean seq635 executed ADR 0004's fixed-boundary attribution on
`router_math_reason_001`. The original and isolated all-Q5 models received
the same 44-token reference prefix and captured 533 tensors each, including
all 40 layer outputs. The original model selected token `25`; Q5 selected
`421`. Layer 0 receives an identical input and is the first nonzero output
delta. Layer 1 is both the first preregistered material amplifier
(`2.08204x` output/input RMSE) and the first router top-k divergence. The
largest incremental boundary RMSE occurs at layer 38 (`0.0200573`).

The evidence rule admitted exactly one candidate: retain the original Q6
tensors through the first material amplifier, layers 0 and 1. Its six exact
Q6 tensors cost `43,008,000` active bytes, below the `64,000,000`-byte cap;
the remaining Q5 lane requires `88.6317 GB/s` at 8k, below the `96 GB/s`
investigation cap.

Clean seq636 tested only that candidate. It converts the other 54 non-head Q6
tensors to Q5 and proves all 639 non-target payloads byte-identical. Traffic
arithmetic passes, but deterministic correctness falls to `2/6` and
`303/429` reference tokens. The original router-math divergence moves from
position 8 to 54, while three previously exact cases acquire divergences.
This is direct evidence that bounded prefix precision islands rearrange
cross-layer error cancellation rather than restore a stable exact boundary.

## Decision

Close `q5_teacher_forced_boundary_attribution_v1` and the complete derived-Q5
replacement/correction family. Do not enumerate another layer prefix, tensor
class, rounding mode, Q4 downgrade, or residual correction. Seq627 and seq628
remain accepted component traffic cuts, not a product engine claim.

Return to the highest-priority offline-repack architecture family. Select one
bounded pre-implementation route:
`q6_sparse_exception_repack_feasibility_v1`.

The gate evaluates one exact, data-dependent representation of each Q6_K
256-value block:

1. choose the optimal contiguous signed 4-bit base window;
2. store its 4-bit codes plus exact position/value exceptions;
3. preserve the original Q6_K scale metadata and prove byte-exact code
   reconstruction;
4. charge block directories, alignment, exceptions, and expert-8 active
   traffic, not only nominal payload bits;
5. derive an optimistic carrier roof from the measured packed-Q4
   `110.522 GB/s`, raw-Q6 `52.7204 GB/s`, and `115 GB/s` planning line.

This is a source-format feasibility gate, not a kernel sweep. It may authorize
one real-full-tensor implementation only if its source-level model predicts
`>=96 GB/s` in original-Q6-equivalent throughput while retaining exact Q6
codes and avoiding persistent I8 expansion. If the optimistic ceiling is
below `96 GB/s`, close the format without implementing it and switch routes.

## Consequences

- Derived Q5 is closed even though its byte arithmetic fits. Correctness
  failure is terminal under the ADR 0004 one-candidate rule.
- Low4/high2 fixed split-plane DPAS v1 remains closed. The new gate is
  admissible only because its residual is data-dependent and sparse; it must
  prove that distinction quantitatively before kernel work.
- Workgroup, API, local-size, exception-threshold, and codec-family sweeps are
  not authorized. The optimal base window is part of this single exact codec.
- A static pass is not a speed claim. A later component kernel would still
  need exact reconstruction, real-tensor numeric comparison, compiled-ISA
  evidence, repeat timing, and the `>=96 GB/s` kill-number.
- Prefill still requires a genuine token-blocked XMX/DPAS GEMM plus chunked
  attention at `<=575.33 us` per 64-token whole layer. It remains the first
  parked alternate if the exact-Q6 format gate fails.
- The project goal remains active; route closure is not product completion.

## Evidence

- `output/q5-boundary-attribution-gate-20260711Tseq635cleanZ/`
- `output/q5-surrogate-feasibility-gate-20260711Tseq636cleanZ/`
- `output/q5-surrogate-feasibility-gate-20260711Tseq631v1cleanZ/`
- `output/q6-splitplane-dpas-gate-20260711Tseq630cleanZ/`

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
