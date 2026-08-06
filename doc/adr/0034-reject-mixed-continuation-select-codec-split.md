# ADR 0034: Reject mixed continuation and select a codec split

Date: 2026-07-11

## Status

Accepted as a deterministic-token rejection and one bounded attribution gate.
The seq687 component and first-distribution evidence remains valid; no product
correctness or speed promotion is implied.

## Context

Clean seq688 retains every seq687 prerequisite: 40/40 live same-state
components pass, aggregate routed relative L2 remains `0.000299383`, and the
first full-vocabulary row remains KLD `0.000369903` with identical top-1 token
`264`.

The injected prefill state does not preserve deterministic continuation. The
baseline greedy IDs are:

`[264, 264, 271, 248068, 198, 8160, 579, 264]`

The injected row produces:

`[264, 264, 264, 264, 264, 264, 264, 264]`

The first divergence is position 2 (zero based), after two identical tokens.
M=1 execution is the same CPU-reference graph in both rows and native
injection is disabled after prefill, so the failure attributes to accumulated
prefill-state movement rather than a native decode implementation.

Evidence:

- `output/all-layer-live-state-injection-tokens-20260711Tseq688cleanZ/`
- `output/all-layer-live-state-injection-20260711Tseq687cleanZ/`

## Decision

Reject the current mixed 20-Q4/20-exact-Q6 carrier as an exact-token R2
prefill state. Preserve its component and first-distribution evidence only.

Run exactly two attribution rows with identical component execution and
continuation gates:

1. compare all 40 same-state native outputs but overwrite only the 20 Q4-down
   layers;
2. compare all 40 same-state native outputs but overwrite only the 20 exact-Q6
   layers.

Each row retains first full-vocabulary KLD `<=0.005`, identical first top-1,
and eight exact greedy IDs. Do not split by individual layer, layer prefix,
threshold, prompt, workgroup, or quantizer until this codec-level result is
recorded.

## Consequences

- If exactly one codec-only row fails, the next correctness repair is confined
  to that carrier's precision boundary.
- If both pass individually but the mixed row fails, the next gate targets
  additive cross-codec precision loss, starting with the shared F16
  contribution/store boundary rather than layer sweeps.
- If both fail, compare their first divergence and distribution movement before
  selecting one precision repair.
- Hybrid wall time remains ineligible for speed claims.

## Follow-Up

- Add a Q4/Q6 application mask to the existing live-injection harness; still
  execute and compare every layer.
- Run one clean Q4-only row and one clean Q6-only row, then record the route
  switch before implementation.

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
