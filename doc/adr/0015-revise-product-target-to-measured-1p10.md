# ADR 0015: Revise the product target to measured 1.10x

Date: 2026-07-11

## Status

Accepted; supersedes ADR 0014 as the active route decision while preserving its
closure of the fixed GPU+NPU architecture. ADR 0039 later raises the absolute
table after correcting raw-prompt OpenVINO measurement; the 1.10x decision
remains in force.

## Context

The owner requires a concrete target that significantly exceeds same-host
OpenVINO Q4 in both phases and is attainable on measured PTL hardware.  The
first quantitative contract selected `1.25x` from the 115 GB/s planning line.
Seq655 later replaced that roof proxy with source-exact native evidence and
proved the only passing 1.25x architecture bound infeasible: the fixed NPU
third limits the pair to `39.376 GB/s` even when GPU time is charged as zero.
ADR 0014 therefore closed every bounded route under the 1.25x contract and
required an explicit owner decision before lowering the target.

The remaining native evidence identifies a narrower target that is both
material and source-reachable:

- The run-to-run noise floor is `0.5%`. A `10%` per-phase advantage is `20x`
  that band and saves about `1.44 s` on the locked 8k/512 denominator row.
- Across 1k-128k, the strict decode inventory at `1.10x` requires
  `96.369-99.585 GB/s`, below the measured `108.793-110.522 GB/s` packed-Q4
  carriers and the 115 GB/s planning line.
- With the accepted traffic cuts, the remaining exact-Q6 lane needs at most
  `57.690 GB/s`. The real raw-Q6 carrier already reaches `52.720 GB/s`, so the
  bounded gap is `9.43%`; the fixed component kill-number is `58 GB/s`.
- At 8k prefill, `2447 tok/s` gives the real layer-27 routed boundary a
  `9771.436 us` cap. Seq652's correctness-safe source-realizable projection is
  `9539.674 us`, leaving `231.762 us` (`2.43%`) implementation headroom.

## Decision

Set minimum product acceptance to the greater of the absolute bucket table and
`1.10x` the same-run OpenVINO median, independently for prefill and decode.
Set `1.125x` as the stretch target. Keep the 256k absolute target at
`400 / 10.00 tok/s` because OpenVINO has no denominator there.

Authorize one dual-phase native route,
`measured_1p10_exact_q6_f16_contribution_v1`, with two pre-registered component
gates:

1. real full-tensor exact Q6_K/Q8_K decode carrier `>=58 GB/s`;
2. seq652's fixed real layer-27 source-realizable grouped/F16-contribution
   implementation `<=9771.436 us`, with the existing all-value gates.

Run the decode kill-number first.  A failure returns to the owner; it does not
authorize layout, workgroup, API, precision, or correctness sweeps.  Passing
both component gates authorizes integration, not a product speed claim.

## Consequences

- The absolute 1k-128k table and machine-readable acceptance contract move to
  version 0.3. Both phases, every bucket, both output lanes, same-run OpenVINO,
  correctness, smoothness, and confirm requirements remain hard gates.
- ADR 0014's NPU and 1.25x-route closures remain in force. No NPU compiler,
  representation, or partition variant is reopened.
- The target may be raised after evidence. Any reduction below 1.10x requires
  another explicit owner revision backed by new hardware evidence.

## Follow-Up

- Prove or reject the fixed `58 GB/s` exact-Q6 carrier on a real full tensor.
- Only after that passes, implement the fixed seq652 prefill dataflow.
- Integrate and run the complete product matrix only after both gates pass.

Evidence:

- `output/npu-exact-q6-representation-20260711Tseq655cleanZ/`
- `output/f16-contribution-plane-feasibility-gate-20260711Tseq652cleanZ/`
- `output/gpu-q6-qmatvec-layer7-ffn-down-full-20260702T234500Z/`
- `output/gpu-q4x8-qmatvec-ffn-gateup-full-20260702T225500Z/`
- `doc/reference/intel-qwen36-35b-a3b-gguf-q4km/performance-target-2026-07-11.md`

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
