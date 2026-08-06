# ADR 0002: R0 256k Prompt-Edge Top-K Policy

Date: 2026-06-26

## Status

Accepted for R0.

## Context

The locked model contract records `context_length=262144`. The materialized
oracle prompt ladder includes exact 262144-token rows for `sentinel_256k` and
`prefill_shape_256k`, and token-id capture verified both rows at exactly
262144 tokens.

The current llama.cpp CPU server top-k route was then run with `n_ctx=262144`
against both exact prompts. Both `/completion` requests returned HTTP 400
`exceed_context_size_error`:

```text
request (262144 tokens) exceeds the available context size (262144 tokens), try increasing it
```

Evidence:

- `output/r0-oracle-prompt-materialization-20260626T082201Z/`
- `output/r0-oracle-token-id-capture-20260626T083347Z/`
- `output/r0-oracle-topk-smoke-20260626T144950Z/`
- `output/r0-oracle-256k-prompt-edge-policy-20260626T145727Z/policy.json`

## Decision

Accept the exact 262144-token first-token top-k row as a prompt-edge policy
case for R0.

The exact prompts remain valid prompt materialization and token-id evidence.
However, first-token prediction after a 262144-token prompt requires one
additional context slot. Under the locked `context_length=262144` contract, the
exact 256k first-token top-k row is policy-resolved as unavailable rather than
captured.

This policy does not create top-k logits, does not close the oracle gate, and
does not authorize a product correctness claim at 262144.

## Consequences

- The latest successful bounded top-k capture remains the 128k artifact.
- Repeating the same exact-context llama.cpp CPU top-k attempt is out of scope
  unless the mechanism changes.
- Raising `n_ctx` beyond 262144 is an over-context diagnostic, not silent R0
  acceptance evidence.
- Future full oracle bundle validation must explicitly encode the 256k
  prompt-edge rows or supersede this ADR with a valid capture route.
- R0 still requires full-ladder teacher-forced distribution references,
  per-boundary input/output tensors, full oracle bundle validation, and
  resident harness `load(model, oracle_bundle)` evidence.

## Follow-Up

- Add explicit 256k prompt-edge handling to the future full oracle bundle
  schema/validator.
- Capture full-ladder teacher-forced distribution references.
- Capture per-boundary reference input and output tensors.
- Load the real oracle bundle through the resident harness.
