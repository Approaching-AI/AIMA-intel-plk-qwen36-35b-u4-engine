# ADR 0070: Adopt OpenVINO U4 specialization runtime

Date: 2026-07-13

## Status

Accepted by owner direction. This supersedes the owner-contract blocker in
ADRs 0048, 0050, 0061, 0068, and 0069 without reopening their rejected native
implementations.

## Context

The locked native-GGUF route has two independent terminal blocks. Its measured
prefill sources cannot fit the 8k `407.968 ms` per-1024-token cap after required
work is charged, and its long-context decode carrier can clear zero-state rate
guards but cannot construct product-correct prompt state. Clean seq798 shows
that OpenVINO U4 and GGUF already differ at short accumulated-state semantics;
clean seq799 shows that exact-GGUF native sequential replay also fails the
full-vocabulary distribution ruler.

OpenVINO itself is not the hardware ceiling. The locked long-context target
requires about `96.416-99.885 GB/s`, while real packed low-bit carriers measure
`108.793-110.522 GB/s`. The OpenVINO 1024-token hidden-body profile also exposes
large, actionable GPU buckets:

| category | profiled time | share |
|---|---:|---:|
| Transpose/data movement | 78.503 ms | 19.06% |
| GatedDeltaNet | 71.428 ms | 17.34% |
| DynamicQuantize | 47.337 ms | 11.49% |
| MoE/MLP | 84.609 ms | 20.54% |

All 30 GatedDeltaNet nodes use `ocl::gated_delta_net::ref___f16`. At 8k, the
stock denominator of `2281.314 tok/s` corresponds to about `448.864 ms` per
1024 input tokens, while the `2510 tok/s` floor permits `407.968 ms`; the
candidate therefore needs an end-to-end reduction of at least `40.896 ms` per
1024-token equivalent before a faster same-run denominator raises the bar.

Evidence:

- `output/openvino-hidden-prefill-profile-20260712Tseq751cleanZ/profile.json`
- `output/reference-state-import-20260713Tseq798cleanZ/`
- `output/native-sequential-semantic-20260713Tseq799cleanZ/`
- `doc/reference/intel-qwen36-35b-a3b-gguf-q4km/performance-target-2026-07-13.md`

## Decision

Adopt the locked OpenVINO U4 IR as the product model and accumulated-state
semantics. Permit OpenVINO GPU and candidate-specific custom OpenCL GPU
operations as final runtime dependencies.

The comparison contract is now:

1. The untouched stock OpenVINO GPU pipeline over the locked IR is both the
   product correctness reference and the performance denominator.
2. The candidate uses the same locked IR and OpenVINO release base, but may use
   explicit custom kernels, derived static graphs, plugin properties, and
   state precision/layout changes.
3. The locked model directory is immutable. Candidate-derived graphs, kernels,
   caches, and configurations live separately and must be enumerated in the
   promotion manifest.
4. Stock and candidate workers are isolated. The stock worker must not load a
   candidate custom-op configuration or share a candidate plugin cache.
5. The hard product target remains unchanged: batch 1, cold no-prefix, exact
   `2k/4k/8k/16k/32k/64k/128k` input, exactly 512 output tokens, and at least
   `1.10x` stock OpenVINO independently in prefill and decode at every bucket.
6. Correctness is candidate versus stock OpenVINO: teacher-forced
   `KLD <= 0.005`, top-1 rate `>=0.99`, exact deterministic greedy tokens, and
   sentinel truth/smoothness across the complete product matrix.
7. llama.cpp/GGUF and the previous native runtime remain diagnostic references
   and hardware evidence. Their disagreement with OpenVINO no longer vetoes a
   product candidate.

The first optimization route, after an immutable-baseline/no-op substitution
gate, is a PTL-specialized GatedDeltaNet plus layout/Transpose bundle. The
route-facing 1024-token component target is to remove at least `40.896 ms` from
the measured Transpose+GatedDeltaNet envelope, followed by an 8k end-to-end
paired prefill pass. Long-context validation still runs `32k/64k/128k` first.

This decision does not claim a speedup, accept the existing hidden-body profile
as end-to-end timing, or allow prefix reuse, prompt lookup, speculative decode,
continuous batching, or a changed output contract to satisfy the product goal.

## Consequences

- The previous owner-decision blocker is resolved; the new open gate is the
  OpenVINO baseline/correctness bootstrap in the specialization roadmap.
- Native route closures stay closed and are not re-run merely because their
  old final-runtime restriction changed.
- The existing stock performance rows remain absolute-floor evidence, but the
  first candidate promotion must refresh the same-run isolated stock baseline.
- Existing GGUF boundary/oracle bundles remain useful diagnostics; they are no
  longer the product correctness authority.
- Decode state must remain inside the OpenVINO stateful pipeline. Cross-runtime
  state import is not part of the new route.

## Follow-Up

- Execute OV0 in the active OpenVINO specialization roadmap.
- After OV0, implement only the pre-registered GatedDeltaNet/layout route; use
  the dynamic-quantization, state-compression, and exact-bucket routes only via
  the recorded direction trigger.
- Supersede this ADR only through another owner-recorded change to model
  semantics, final runtime, product matrix, or required OpenVINO ratio.

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
