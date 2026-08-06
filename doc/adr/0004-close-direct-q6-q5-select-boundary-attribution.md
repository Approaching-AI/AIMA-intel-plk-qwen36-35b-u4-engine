# ADR 0004: Close direct Q6/Q5 replacement and select boundary attribution

Date: 2026-07-11

## Status

Superseded by ADR 0005 after the single admitted correction failed the
six-case correctness gate.

## Context

The product contract remains at least `1.25x` same-host OpenVINO GPU U4 in
both phases at every bucket. The headline 8k decode floor is `52.79 tok/s`,
which leaves `18.94298 ms/token`. After the accepted head and router
exact-refine traffic cuts, the remaining Q6 lane has `3.74439 ms/token`.

ADR 0003 required one exact compact-Q6 carrier to reach `>=96 GB/s`. Clean
seq630 tested the prescribed low4/high2 split-plane DPAS carrier on the real
`blk.7.ffn_down_exps.weight` tensor. It passed the numeric gate at maximum
absolute error `1.192092896e-7` and compiled to DPAS, but reached only
`17.2478 GB/s`. That is slower than the existing raw-Q6 `52.7204 GB/s`
carrier and far below the kill-number. The branch therefore hit its recorded
stop condition before any tuning sweep.

A derived Q5_K replacement is arithmetically small enough: converting all 60
non-head Q6 tensors reduces their active bytes from `352,665,600` to
`295,567,360`. Keeping the `26,880`-byte exact-Q6 head-refine rows requires a
Q5 carrier of `78.9469 GB/s` at the worst 8k bucket, so `80 GB/s` was the
pre-registered promotion line.

The clean precision gates reject direct substitution:

- seq631 changes exactly those 60 tensors and proves all 633 other tensor
  payloads byte-identical. It passes the traffic arithmetic but only `5/6`
  deterministic cases; `router_math_reason_001` first changes at generated
  position 8 (`25` to `421`).
- seq632 converts only the 40 Q6 FFN tensors and keeps attention Q6. It passes
  only `3/6` cases, and the retained raw-Q6 lane alone exhausts the target
  budget.
- seq633 converts only the 20 Q6 attention tensors and keeps FFN Q6. It passes
  only `2/6` cases and would require `205.438 GB/s` from Q5 at 8k.

The class results are non-monotonic: converting both classes passes more cases
than either class alone. Class or layer enumeration is therefore not a valid
attribution method.

## Decision

Close `surrogate_refine_splitplane_dual_phase_v1` as an active architecture.
Preserve the seq627 Q4-head/exact-Q6-top16 and seq628 I8-router/exact-F32-top16
cuts as accepted components, but do not integrate them into a product claim.

Close both direct successors:

1. exact low4/high2 compact-Q6 split-plane DPAS v1;
2. uniform or classwise non-head Q6-to-Q5 substitution.

Select `q5_teacher_forced_boundary_attribution_v1` as the sole active
investigation route. It is not yet an engine architecture. Its first and only
open unit is the seq631 failure boundary:

1. Teacher-force the original reference prefix through generated position 7
   of `router_math_reason_001` in the original and isolated all-Q5 models.
2. Capture every parameterized layer input and output for the next-token
   logits, then report per-layer delta-in, delta-out, and amplification.
3. Recompute Q6-vs-Q5 local operators from the same captured input at the
   first material amplification boundary. This must distinguish a source
   perturbation from downstream sensitivity.
4. Propose at most one correction/precision-island bundle from that evidence;
   do not enumerate layer subsets or quantization variants.

Any proposed correction must preserve the 8k byte/time budget before it is
implemented. At a Q5 carrier of `80 GB/s`, only `5.804 MB` of active Q6 can be
retained. The absolute investigation cap is `64.0 MB` of raw-Q6-equivalent
active correction traffic and a derived Q5 requirement of `<=96 GB/s`; at
`96 GB/s`, the arithmetic maximum is `64.960 MB`. A different residual format
must be charged by its measured bytes and carrier time against the same
`3.74439 ms` lane.

## Consequences

- No compact-Q6 workgroup, API, split-plane, or ordinary expanded-I8 sweep is
  admissible from seq630.
- No Q4 downgrade, class split, layer subset, or Q5 rounding sweep is
  admissible from seq631-633 without the fixed-boundary source attribution.
- A bounded correction must first recover all six short/router deterministic
  sequences, then pass the locked teacher-forced distribution ladder. That is
  still correctness evidence, not a native speed claim.
- If the single source-attributed correction exceeds `64.0 MB`, requires more
  than `96 GB/s` from Q5, or fails the six-case gate, close the derived-Q5
  family and return to architecture selection.
- Prefill still requires a genuine token-blocked XMX/DPAS GEMM plus chunked
  attention. Its `575.33 us` per 64-token whole-layer ruler is unchanged, but
  decode attribution is the current open gate.

## Evidence

- `output/q6-splitplane-dpas-gate-20260711Tseq630cleanZ/`
- `output/q5-surrogate-feasibility-gate-20260711Tseq631v1cleanZ/`
- `output/q5-surrogate-feasibility-gate-20260711Tseq632cleanZ/`
- `output/q5-surrogate-feasibility-gate-20260711Tseq633cleanZ/`

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
