# ADR 0043: Reject register recurrence and select one chunk-64 WY gate

Date: 2026-07-12

## Status

Accepted for the next native-prefill design gate. This does not accept a
linear-attention component or any product-speed row.

## Context

The clean same-host OpenVINO hidden-prefill profile attributes `71.428 ms` to
the 30 GatedDeltaNet nodes at 1024 tokens. The earlier calibration row was
`72.088 ms`; its mean divided by the product `1.10x` ratio pre-registered a
conservative `2184 us/layer` state-core cap.

Clean seq753 evaluates a fixed F32 register-resident recurrence over real
layer-0 1024-token boundaries. Attention, final state, and normalized output
all pass the locked cosine/relative-L2 thresholds, and the compiler reports
SIMD32, 256 GRFs, four EU threads, and no spill. It does not stably clear the
cap: repeat/confirm are `2211.041 / 2166.770 us`, one row misses, and paired
spread is `2.043%` versus the `0.5%` noise band.

OpenVINO's measured GDN binary uses half arguments, SIMD16, 128 GRFs, and eight
EU threads. Seq754 therefore admits exactly one matching-storage design rather
than a workgroup sweep. That design is decisively rejected:

- repeat/confirm state-core time is `5326.041 / 5319.270 us` (`0.127%` spread);
- attention relative L2 is `0.030436` and final-state relative L2 is
  `0.089327`, both above `0.002`;
- the normalized final output happens to pass at `0.001447`, but cannot erase
  the failed state boundaries;
- the compiler still allocates 256 GRFs and four EU threads, so the intended
  128-GRF/eight-thread resource shape is not obtained.

Evidence:

- `output/openvino-hidden-prefill-profile-20260712Tseq751cleanZ/`
- `output/linear-attention-prefill-boundary-20260712Tseq750cleanZ/`
- `output/linear-attention-prefill-state-20260712Tseq753cleanZ/`
- `output/linear-attention-prefill-f16-state-20260712Tseq754cleanZ/`

## Decision

1. Close the per-token register-resident F32 mapping as an accepted prefill
   component: it is numerically valid but does not repeatably clear its hard
   cap inside the registered noise band.
2. Close half-state recurrence and all workgroup/column/subgroup variants of
   that mapping. Its timing, state accuracy, and compiler-resource axes all
   fail.
3. Admit one algorithmic successor: the fixed 1024-token, chunk-size-64 WY
   representation used by the established chunked gated-delta formulation.
   It has exactly 16 chunks, keeps the recurrent state and accumulations F32,
   may use half storage for Q/K/V and chunk intermediates, and must not expose
   chunk-size, precision, workgroup, or subgroup sweep flags.
4. Time the complete chunk GDN core -- gate cumulative sums, lower-triangular
   WY solve, W/U construction, 16-chunk state scan, and attention output --
   with zero host transfer/readback. Repeat and confirm must each be
   `<=2184 us`, their medians must be within `0.5%`, and real attention/state
   boundaries must retain cosine `>=0.999` and relative L2 `<=0.002`.

The separate final normalization remains a diagnostic until it is fused or
charged into a complete linear-attention tile. Passing this gate would still
not establish whole-layer or product prefill speed.

## Consequences

- Do not rerun the F32 decay-hoist cut, half-state recurrence, or alternate
  register workgroup shapes.
- Chunk-64 is an algorithm change, not permission for an implementation
  matrix. A failure returns to route reflection; it does not authorize
  chunk-16/32, storage-precision, or local-size sweeps.
- Seq753 remains useful as the exact F32 oracle carrier. Seq754 remains a
  closed resource-parity diagnostic and is not an accuracy denominator.

## Follow-Up

- Clean seq755 on commit `9a85741` completes the fixed pipeline. Every real
  boundary passes, and repeat/confirm is stable at `24516.666 / 24577.292 us`
  (`0.247%` spread), but it is `11.23x` the `2184 us` cap. Stage medians are
  prepare `11683.333 / 11700.625 us`, scan `3196.979 / 3199.479 us`, and
  output `9642.708 / 9677.916 us`. The scan also spills 128 bytes.
- Therefore close scalar-FMA chunk-64, keep its correctness implementation as
  an oracle/operation census, and return to route reflection. The next
  admissible action is a separately pre-registered matrix-engine feasibility
  gate, not a chunk/storage/local-size modification of seq755.
- Clean seq756 executes that fixed F16-F16-F32 feasibility gate. Its real-FP16
  numeric comparison passes at relative L2 `0.0001234` and cosine above
  `0.99999999`; compiler evidence is spill-free SIMD16/64-GRF and contains
  exactly eight DPAS instructions. Repeat/confirm is only
  `3.071 / 3.633 TMAC/s` and has `18.294%` spread, so both the `4.0 TMAC/s`
  and `0.5%` checks fail. Close XMX chunked GDN and do not sweep tile shape.

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
