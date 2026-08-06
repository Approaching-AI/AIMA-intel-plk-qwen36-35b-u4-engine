# ADR 0044: Close XMX chunked GDN and select a whole-linear-stage gate

Date: 2026-07-12

## Status

Accepted for the next native-prefill design gate. This does not accept any
linear-attention component or product-speed row.

## Context

Seq756 is the terminal pre-registered XMX feasibility result. The fixed
`65,536 x 8x16x128` F16-F16-F32 tile uses real layer-0 FP16 inputs, compiles to
SIMD16 with 64 GRFs, no spill/scratch, and exactly eight DPAS instructions.
Numeric relative L2 is `0.0001234` with cosine above `0.99999999`. Timing still
fails decisively: repeat/confirm is `3.071 / 3.633 TMAC/s` versus the required
`4.0`, and `18.294%` spread exceeds the `0.5%` noise band.

The state-only `2184 us` allocation is conservative rather than the product
contract itself. The same clean OpenVINO profile attributes these complete
1024-token linear-stage categories across 30 layers:

- projections, normalization, and activation: `50.171 ms`;
- convolution and its reorder: `12.118 ms`;
- GatedDeltaNet: `71.428 ms`.

Their sum is `133.717 ms`, or `4457.233 us/layer`. The product ratio makes the
complete-stage cap `4457.233 / 1.10 = 4052.030 us/layer`. Conservatively
charging the slower seq753 F32 recurrence row (`2211.041 us`) leaves
`1840.989 us`; the registered non-state gate rounds down to `1840 us`.

Seq750 already captures Q/K/V, control, convolution, state, normalized output,
and final output-projection boundaries, but not the normalized hidden input
needed to validate the input projections.

## Decision

1. Close XMX chunked GDN. Do not lower the `4.0 TMAC/s` floor or vary DPAS
   tile, chunk size, precision, subgroup, or workgroup shape.
2. Admit one budget-reallocation route at a materially larger boundary, not a
   state-kernel retry. First capture the real layer-0 normalized-hidden input.
3. Implement one parameterized 1024-token native non-state tile covering input
   projections, convolution/control formation, final normalization, and output
   projection, with zero timed host transfer/readback.
4. Require every captured boundary to pass cosine `>=0.999` and relative L2
   `<=0.002`; repeat and confirm must each be `<=1840 us` and within `0.5%`.
5. Only a pass may attach the unchanged seq753 F32 recurrence and test the
   complete linear tile at `<=4052 us` twice. A pass there supersedes only the
   internal state-budget allocation; it does not relabel seq753 as a passing
   state component.

## Consequences

- Register recurrence, half-state recurrence, scalar chunk-64, and XMX
  chunked-GDN implementation variants remain closed.
- OpenVINO/oneDNN may generate native programs offline, but the runtime must
  link and map neither dependency.
- A non-state miss returns to product-level route reflection. It does not
  authorize projection, convolution, precision, or tile sweeps.
- Even a complete-layer pass remains component evidence until all 30 linear
  layers are integrated and the product prefill/correctness/context matrix is
  measured.

Evidence:

- `output/openvino-hidden-prefill-profile-20260712Tseq751cleanZ/`
- `output/linear-attention-prefill-state-20260712Tseq753cleanZ/`
- `output/f16-dpas-prefill-feasibility-20260712Tseq756cleanZ/`

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.

## Follow-Up

- Clean seq757 captures `attn_norm-0` with the other 17 real layer-0
  boundaries and confirms the `4052.030 / 1840.000 us` whole/non-state caps.
- Clean seq758 evaluates the fixed five-projection subset before convolution,
  control, or final normalization is charged. Repeat/confirm is
  `2036.839 / 2027.162 us` with `0.477%` spread, already above `1840 us`; the
  only fast Q6 K32 carrier also has QKV relative L2 `0.002080` versus `0.002`.
- Close whole-linear budget reallocation and every projection representation/
  codegen variant. Return to product-level budget reflection; seq758 is not a
  native runtime carrier or product-speed row.
