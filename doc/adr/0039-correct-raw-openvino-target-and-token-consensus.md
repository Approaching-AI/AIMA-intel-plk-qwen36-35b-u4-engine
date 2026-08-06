# ADR 0039: Correct the raw OpenVINO target and token consensus

Date: 2026-07-11

## Status

Accepted; amends ADR 0015's absolute table while preserving its 1.10x ratio

## Context

Clean seq723 evaluates the identical first 1,024 token IDs without prefix
caching or chat-template insertion. The reference runtimes disagree at
zero-based generated position 2:

- llama.cpp: `[264,264,271,248068,198,8160,579,264]`;
- OpenVINO GPU U4: `[264,264,264,264,264,264,264,264]`.

No candidate can exactly equal both rows. Treating that disagreement as a
candidate failure makes the multi-reference acceptance predicate impossible.
The native injected row is separately recorded as
`[264,264,264,264,264,264,271,248068]`; it is not promoted.

The same check exposed a performance-protocol error. The first 512-token
OpenVINO matrix counted the raw prompt before generation but left
`GenerationConfig.apply_chat_template=true`, so the timed input was not the
claimed raw token payload. Seq724-726 disable the template, require the runtime
input-token count to equal the bucket, disable prefix caching, and take three
post-warmup medians. The corrected filler row is faster at short contexts, so
the 1.10x absolute target must rise rather than inherit the understated table.

Evidence:

- `output/cross-reference-token-consensus-20260711Tseq723cleanZ/`
- `output/r0-openvino-denominator-matrix-20260711Tseq724-raw-both-cleanZ/`
- `output/r0-openvino-denominator-matrix-20260711Tseq725-raw-filler-cleanZ/`
- `output/r0-openvino-denominator-matrix-20260711Tseq726-raw-filler-rest-cleanZ/`
- `doc/reference/intel-qwen36-35b-a3b-gguf-q4km/performance-target-2026-07-11.md`

## Decision

1. OpenVINO product comparisons use raw prompts with
   `apply_chat_template=false`, prefix caching disabled, and the pipeline's
   reported input-token count equal to the requested bucket.
2. Raise the absolute table to the next whole prefill token/s and next 0.01
   decode token/s above 1.10x the harder corrected/prior denominator. The new
   headline 8k/512 target is `2510 / 46.60 tok/s`, TPOT `<=21.46 ms`, total
   latency `<=14.25 s`.
3. Exact greedy candidate scoring requires reference consensus first. A case
   where llama.cpp and OpenVINO disagree is diagnostic—not a candidate pass or
   failure. At least three consensus prompt cases still require exact match and
   first divergence still blocks promotion.

This does not relax component numeric checks, shared-prefix distribution
checks, sentinel retrieval, smoothness, or same-run OpenVINO comparison.

## Consequences

- Acceptance contract version becomes 0.4. The hard ratio remains 1.10x and
  the stretch ratio remains 1.125x.
- Corrected filler rows cover 1k-128k; corrected prefill-shape/sentinel rows
  currently cover 1k/8k. Remaining raw 512 rows and the 1024-output lane must
  be refreshed before a product claim and may only raise the threshold.
- The corrected 8k routed-boundary cap is `9526.177 us`. Existing historical
  measurements retain their original cap labels; new promotion gates use the
  corrected cap.

## Follow-Up

- Complete raw prefill-shape/sentinel 2k-128k denominators and the 1024-output
  lane before product promotion.
- Build a three-case llama/OpenVINO consensus token suite.
- Return to native end-to-end integration only against the corrected table.

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
