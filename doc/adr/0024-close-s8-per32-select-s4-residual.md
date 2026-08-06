# ADR 0024: Close S8 per-32 alone and select one S4 residual

Date: 2026-07-11

## Status

Accepted as a component-route rejection and one bounded residual successor.
No teacher-forced, token, context-ladder, or product promotion is implied.

## Context

ADR 0023 selected the one scale granularity the measured K32 DPAS codegen can
execute correctly: symmetric S8 with one F32 scale per 32 weights. Clean
seq677 confirms the generated core matches the host-repacked calculation within
`1.160e-7` and clears the `4316.404 us` raw gate at
`3008.195 / 3032.207 us`.

The carrier alone fails the locked accuracy contract. Its weight
requantization has relative L2 `0.00538186`; both full `16,777,216`-value output
comparisons have cosine `0.99998747` and relative L2 `0.00500705`, versus the
required `<=0.002`. The paired timing rows leave only
`1308.209 / 1284.197 us` under the fixed raw gate, so a successor must correct
the representation and fit that measured residual budget.

Evidence:

- `output/onednn-grouped-q6-s8-per32-gate-20260711Tseq677cleanZ/`
- `output/onednn-grouped-q6-exact-preflight-gate-20260711Tseq676cleanZ/`

## Decision

Close symmetric S8-per-K32 as a standalone Q6 down carrier. Select exactly one
residual representation: retain seq677's fixed S8 main plane, quantize its
dequantization residual to one symmetric signed-S4 plane with one F32 scale per
K32, and execute the two offline-generated grouped cores with the residual core
summing into the main F32 destination.

The paired combined-core gate remains `<=4316.404 us`. It must compare all
`16,777,216` outputs at finite, cosine `>=0.999`, and relative L2 `<=0.002`.
The S4 rule is fixed to max-absolute scaling over K32 with codes `[-7, 7]`.

This does not authorize scale/group/tile/workgroup variants, runtime
oneDNN/OpenVINO linkage, or a product speed claim.

## Consequences

- S8-per-K32 remains a measured fast main plane but cannot be promoted alone.
- The S4 residual has at most the `1284.197 us` slower-row headroom before any
  complete-runtime integration; a larger raw cost is terminal by arithmetic.
- A passing offline composition still requires binary extraction and the same
  paired gate in the pure OpenCL runtime.

## Follow-Up

- Run one fixed S8-main plus S4-residual layer-39 paired gate.
- If it passes, extract both kernels from one offline-generated program and
  integrate the pure-OpenCL boundary. If it fails accuracy or the combined raw
  cap, record the terminal gap before changing representation family.

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
