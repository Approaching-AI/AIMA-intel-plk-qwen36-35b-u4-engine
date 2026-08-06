# ADR 0069: Audit native sequential GGUF semantic state

Date: 2026-07-13

## Status

Accepted and completed. Clean seq799 rejects native sequential semantic-state
construction at the pre-registered short distribution gate; the 2k and longer
successors were not run. ADR 0070 subsequently resolves the owner-contract gate
by adopting locked OpenVINO U4 product semantics; this native route stays closed.

## Context

Clean seq798 rejects OpenVINO U4 state import because its state does not share
the locked GGUF Q4_K_M accumulated semantics. The corresponding rejected-route
record permits a successor only when state is constructed from the exact locked
GGUF semantics (or after an owner-recorded contract change).

The existing packed Level Zero backend already executes the accepted
parameterized forty-layer GGUF token program and updates all native linear and
full-attention state on every submission. Its default short gate proves
CPU-constructed prompt state plus native decode, while seq793-795 prove only
zero-state long-context capacity. Submitting the actual prompt tokens from
position zero is therefore the smallest way to isolate whether the same native
program can construct prompt-conditioned state. It changes no backend kernel or
model arithmetic.

Evidence:

- `output/packed-token-level-zero-backend-20260713Tseq793-int8-hot8192-tile4-hostucb-cleanZ/result.json`
- `output/packed-token-context-gap-20260713Tseq795-int8-hot8192-tile4-hostucb-output512-cleanZ/result.json`
- `output/reference-state-import-20260713Tseq798cleanZ/result.json`
- `doc/adr/0068-reject-reference-state-import-restore-owner-decision.md`

## Decision

Add one parameterized `--sequential-prompt` diagnostic to the existing packed
backend smoke and gate it in stages:

1. Start from zero native state and submit every exact prompt token in order.
2. On the locked short `fresh_code_03` case, require the existing full-vocabulary
   CPU GGUF ruler at every predicted position: maximum KLD `<=0.005` and top-1
   rate `>=0.99`, plus exact reference token IDs.
3. Only if the short mechanism passes, run `sentinel_002k` and
   `prefill_shape_002k` against the accepted llama.cpp oracle. Require exact
   teacher-forced top-1 IDs; exactness proves deterministic greedy equivalence
   by induction. The sentinel answer must occur inside the exact matched prefix
   of the accepted reference response.
4. Only if 2k passes, extend the same unchanged mechanism to the route-priority
   `32k/64k/128k` sentinel and prefill-shape cases. Stop at the first semantic
   failure; do not sweep prompt, state, codec, hot-window, or kernel axes.

Sequential prompt throughput and TTFT are recorded so this route cannot be
mistaken for product prefill. They are diagnostic only and are expected to miss
the product prefill floors by a wide margin. Decode timing is also diagnostic
until the complete product protocol is run.

## Consequences

- A pass may establish prompt-conditioned native decode correctness and remove
  the zero-state semantic ambiguity.
- A pass does not reopen ADR 0048/0050 native-prefill performance sources, meet
  a product row, allow a speedup claim, or permit OpenVINO in the final runtime.
- A short or 2k failure closes this diagnostic without variants and restores
  ADR 0068 unchanged.
- Even a complete semantic ladder leaves the owner-contract/new-capability gate
  active for product prefill and full matrix promotion.

## Outcome

Clean seq799 at commit `3ef5077` starts from zero state and submits all 24
`fresh_code_03` prompt tokens through the unchanged packed backend. Its nine
teacher-forced predictions match the exact locked-GGUF CPU top-1 IDs, so greedy
token equivalence alone would appear to pass. The full-vocabulary ladder does
not: maximum KLD is `0.0233093327792`, with the second prediction already at
that maximum, versus the locked `0.005` limit. Top-1 rate is `1.0`.

The result is a semantic failure, not noise or a performance result. Sequential
state construction measured only `33.660 tok/s` and is explicitly ineligible
as product prefill. Per the terminal staging rule above, no 2k sentinel,
prefill-shape, longer-context, alternate-prompt, state-codec, or kernel variant
is admitted.

Evidence:

- `output/native-sequential-semantic-20260713Tseq799cleanZ/`

## Follow-Up

- Keep seq799 as the terminal native sequential-state audit.
- Resume product work only through the unchanged owner-contract/new-capability
  gate; do not reinterpret 9/9 top-1 agreement as distribution correctness.

Current product gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
