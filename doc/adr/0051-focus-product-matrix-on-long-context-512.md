# ADR 0051: Focus the product matrix on long-context output-512

Date: 2026-07-13

## Status

Accepted by owner direction. The project goal remains active.

## Context

The previous product contract crossed ten input buckets from 1k through 256k
with both 512- and 1024-output lanes. That breadth made short-context evidence
and prompt-edge coverage part of the same completion claim as the use case the
owner now prioritizes: sustained batch-1 inference after a long prompt with a
meaningful 512-token answer.

Clean seq743/744 already pass a scoped 1k decode row, but there is no promoted
native prefill engine or long-context product ladder. Treating that short row
as the active headline would optimize a diagnostic that cannot answer the
long-context goal. Conversely, the measured 100k/256k and 1024-output artifacts
remain useful for diagnosis and should not be discarded.

## Decision

Make the sole promotion matrix:

- exact input tokens: `2048`, `4096`, `8192`, `16384`, `32768`, `65536`, and
  `131072`;
- exact output tokens: `512`;
- batch 1, resident model, cold no-prefix operation;
- independently in every bucket and phase,
  `max(absolute target, 1.10 * same-run OpenVINO median)`;
- stretch ratio `1.125x`, with no averaging across buckets or phases.

The `1024`, `102400`, and `262144` input buckets and every 1024-output row are
diagnostic/non-gating. Their prompt assets and historical artifacts stay in the
repository. Route selection should profile the 32k/64k/128k bottlenecks first,
while 2k through 16k remain mandatory regression guardrails.

This is a scope change, not a ratio, correctness, or hardware relaxation. The
unchanged 8k target means ADRs 0048 and 0050 still close the GPU/NPU prefill
families they measured. A new source route must remain materially independent
and clear a complete target-facing bound before implementation.

## Consequences

- Seq743/744 remain accepted engineering evidence but no longer satisfy a core
  product row because 1k is outside the matrix.
- Product denominator refresh now needs only output-512 replay for the missing
  `prefill_shape` and `sentinel` rows at 2k/4k/16k/32k/64k/128k.
- Completion requires seven input rows, both phases, correctness, sentinel,
  smoothness, memory, and no-OOM evidence.
- 100k, 256k, and 1024-output failures cannot block promotion; successes there
  cannot compensate for a failing core row.
- The next target-facing reconciliation is the core denominator refresh and a
  current-native 2k-128k/512 context-ladder gap map. It may select a new route
  but does not reopen a rejected source family by itself.

Authority:

- `goals/intel-qwen36-35b-a3b-q4km-engine.md`
- `benchmarks/intel-qwen36-35b-a3b-gguf-q4km/acceptance-matrix.json`
- `doc/reference/intel-qwen36-35b-a3b-gguf-q4km/performance-target-2026-07-13.md`
