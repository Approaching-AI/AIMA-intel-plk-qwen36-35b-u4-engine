# ADR 0049: Reopen one prefill-only GPU+NPU complete-FFN gate

Date: 2026-07-12

## Status

Superseded by ADR 0050. Its bounded phase-split rationale and no-sweep contract
remain the authority for interpreting seq767.

## Context

ADR 0048's route inventory treated the rejected fixed GPU+NPU architecture as
one indivisible dual-phase route. Clean seq766 identifies later evidence that
makes a prefill-only phase split materially different:

- clean seq743/744 now provide a native GPU-only short-decode repeat/confirm at
  `50.433 / 50.500 tok/s`, so the failed NPU M=1 decode carrier is no longer on
  the candidate prefill path;
- seq759 supplies a `626.566 us` fused router/scatter shell that did not exist
  at the ADR-0014 stop;
- seq762 restores the shared expert and final `ffn_out` boundary.

Applying seq653v2's target-measured concurrent M64 rate (`11.9014 TOPS`) to
seq759's routed M8 padding plus the shared expert gives `63.821B` complete
operations and a `5362.447 us` matrix projection. Adding the fused shell and
seq764's measured `117.395 us` shared scalar gate projects
`6106.408 us` versus the `6250 us` cap. The `143.592 us` margin exceeds two
registered `0.5%` noise bands (`62.5 us`). The required aggregate rate is
`11.591 TOPS`.

This projection is intentionally optimistic and not promotable. Seq653v2's
proxy spread is `16.676%`; exact Q4_K affine work, cross-device synchronization,
shared final add, and queue drains are not independently charged. Those facts
require a complete measured source gate rather than justify a speed claim.

## Decision

Admit exactly one
`gpu_npu_prefill_only_exact_q4_complete_ffn_component_v1` source gate:

1. Use the fixed `2:1` GPU:NPU disjoint routed-row partition; keep the shared
   expert on GPU. Do not sweep the partition.
2. Consume the real layer-27, 1024-token assignments and all seq763 boundary
   tensors through final `ffn_out`.
3. Preserve exact GGUF Q4_K group-32 scale and affine-min semantics; include
   routed/shared gate/up, SwiGLU, down, weighting, scalar gate, and final add.
4. Use native Level Zero GPU plus NPU graph ABI. Offline compiler use is
   allowed, but the measured executable may map no OpenVINO/oneDNN runtime.
5. Charge shared allocation, fences, cross-device synchronization, queue
   drains, router/gather/compact assignment, and deterministic scatter. Timed
   host upload/readback must both be zero.
6. Require cosine `>=0.999`, relative L2 `<=0.002`, complete repeat/confirm
   `<=6250 us`, and paired spread `<=0.5%`.
7. Any legality, build, correctness, cap, or noise failure closes the source.
   No compiler flag, graph shape, representation, precision, bucket, partition,
   or synchronization sweep is authorized.

This decision reopens neither NPU decode nor the old exact-Q6 output-head
carrier. It corrects ADR 0048's incomplete phase inventory without changing
hardware, model, correctness, batch, native-runtime, or `1.10x` contracts.

## Consequences

- The project is no longer waiting on an owner contract change; one bounded
  implementation route is active.
- Seq766 is design arithmetic only. Product prefill, context, smoothness, and
  the complete matrix remain open.
- A source-gate failure restores ADR 0048's owner-decision state immediately;
  it does not authorize tuning.

Evidence:

- `output/phase-split-gpu-npu-prefill-reopen-20260712Tseq766cleanZ/`
- `output/product-architecture-feasibility-reconciliation-20260711Tseq653v2cleanZ/`
- `output/packed-token-level-zero-real-backend-gate-20260712Tseq743-distribution-cleanZ/`
- `output/fused-expert-ffn-design-gate-20260712Tseq759cleanZ/`

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
