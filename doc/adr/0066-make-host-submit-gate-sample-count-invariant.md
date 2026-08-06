# ADR 0066: Make the host-submit gate sample-count invariant

Date: 2026-07-13

## Status

Accepted; refines how ADR 0042's unchanged `0.100 ms` host budget is tested.

## Context

The clean hot8192 output-512 seq792 run clears the decode-rate 95% upper-bound
cap in all seven core buckets. Its aggregate gate nevertheless fails because
one of 512 submit calls at 16k measures `0.314566 ms`; the other six bucket
maxima are `0.021152..0.059913 ms`. The 16k wall median/UCB is
`22.246026/22.258498 ms`, comfortably below its `23.288309 ms` cap.

The backend smoke had interpreted ADR 0042's `0.100 ms` budget as a hard
maximum over every observed host call. That statistic is not sample-count
invariant: a 512-token, seven-bucket run gets 3,584 chances to observe an OS
scheduling interruption, while a twenty-token guard gets only 140. It therefore
rejects longer evidence more often even when the runtime path and typical host
cost are unchanged. The wall distribution already includes such interruptions.

Evidence:

- `output/packed-token-context-gap-20260713Tseq791-int8-hot8192-tile4-core-cleanZ/result.json`
- `output/packed-token-context-gap-20260713Tseq792-int8-hot8192-tile4-output512-cleanZ/result.json`
- `doc/adr/0042-correct-packed-token-state-census-and-budget.md`

## Decision

1. Keep ADR 0042's `0.100 ms` host-submit budget unchanged.
2. Make the C++ smoke responsible for structural invariants and finite timing
   samples. Emit the complete host-submit sample vector plus median, p95, and
   maximum; retain the maximum as a diagnostic only.
3. For promotion runs with at least twenty samples, test the host budget with
   the same deterministic one-sided 95% percentile-bootstrap median upper bound
   used for the wall-latency cap.
4. Preserve the wall-latency confidence gate. Passing the host decomposition
   cannot compensate for a wall-rate miss.
5. Keep seq792 as failed pre-change evidence and rerun the full lane on a clean
   commit. Do not reinterpret or mutate its artifact.

This decision does not promote semantic correctness or authorize a product
speed claim.

## Consequences

- A persistent host regression above `0.100 ms` still fails, while an isolated
  scheduler interruption remains visible without vetoing an otherwise stable
  distribution.
- Sub-twenty-sample probes remain diagnostics for this host inference; they
  cannot establish the host budget statistically.
- The full output-512 lane must be collected again after this instrumentation
  change before the capacity gate can close.

## Follow-Up

- Rebuild and test the smoke and confidence helper.
- Commit the instrumentation change, then rerun correctness, twenty-token, and
  output-512 evidence at that exact clean commit.

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
