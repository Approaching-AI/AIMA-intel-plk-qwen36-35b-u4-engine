# ADR 0013: Select a parameterized GPU+NPU exact component gate

Date: 2026-07-11

## Status

Accepted; supersedes ADR 0012 as the active route decision.

## Context

ADR 0012 required one product-level CPU/GPU/NPU architecture reconciliation
after the grouped exact-Q4_K prefill family was exhausted. Clean seq653v2
performs that audit on the locked PTL host and records paired distributions,
not nominal device specifications.

The independent single-device routes do not clear the product contract:

- CPU native decode is `4.2 tok/s` versus the headline `52.79 tok/s` floor.
- GPU-only exact grouped prefill remains closed at `11389.091 us` minimum
  versus the `8514.926 us` layer-27 routed-MoE cap.
- The NPU reports `25.1904 TFLOPS` FP16 and `50.3808 TOPS` INT8, and
  `query_model` assigns all `16,051 / 16,051` language-model operations to
  it. The complete VLM graph nevertheless fails after `193.353 s` in the NPU
  compiler's location verifier with `11,520` duplicated names. This is not an
  OOM: the compile process remained near `4.5 GiB` RSS with about `54 GiB`
  system memory available.
- A parameterized real-weight NPU graph does compile and execute, but NPU-only
  M=64 performance projects the complete prefill boundary to `16846.975 us`,
  nearly twice the cap.

The materially independent result is concurrent GPU+NPU partitioning. Seq653v2
extracts the real `508,559,360`-byte OpenVINO language-model head as a fixed
M=1 and M=64 compiler/hardware proxy and runs one full proxy concurrently on
each device. The proxy is deliberately matched to the arithmetic intensity of
the real routed-MoE matrix work. Numeric comparisons pass:

- NPU M=1 versus CPU: cosine `0.9999979734`, max abs `0.00555694` over all
  `248,320` outputs;
- NPU M=64 versus GPU: cosine `0.9999995828`, max abs `0.00542426` over all
  `15,892,480` outputs.

Paired medians are `9438.144 us` at M=1 and `10939.167 us` at M=64. They
produce two complete architecture bounds:

1. M=1 aggregate streaming is `107.7668 GB/s`, above ADR 0003's exact-Q6
   `96 GB/s` kill-number. Combining that carrier with the accepted Q4-head /
   exact-Q6-top16 and I8-router / exact-F32-top16 cuts, measured packed-Q4
   `110.522 GB/s`, the 8k KV stream, and the preregistered `450 + 350 us`
   router/schedule budgets projects `18567.940 us/token`, or `53.856 tok/s`,
   versus the `18942.982 us` / `52.79 tok/s` headline gate.
2. M=64 aggregate compute is `11.9014 TOPS`. Charging the exact layer-27
   `51,539,607,552` matrix operations, the measured multiple-of-8 routing
   padding (`8192 -> 9120` assignments across 26 buckets), and seq650's full
   `3444.096 us` gather/residual/activation/weight/scatter shell projects
   `8265.224 us`, below the `8514.926 us` cap. The matrix-only cap is
   `5070.830 us`, requiring at least `11.3153 TOPS` aggregate.

These are narrow feasibility margins, not native results. The OpenVINO graph
is a compiler/hardware probe only and cannot be linked by the promoted runtime.

## Decision

Select exactly one successor:
`gpu_npu_parameterized_exact_q6_variable_m_component_v1`.

The component gate is fixed before implementation:

1. Use a `2:1` GPU:NPU disjoint row/work partition, derived from the paired
   solo M=1 and M=64 device rates. Do not sweep the partition.
2. For decode, consume a real GGUF Q6_K tensor and real Q8_K input, preserve
   exact Q6 code/scale semantics, compare every output against the existing
   native oracle, and measure concurrent raw-Q6-equivalent throughput. The
   paired median must be at least `96 GB/s`.
3. For prefill, consume seq639's exact layer-27 assignments and seq646's
   captured inputs/oracles. Use the fixed 26 multiple-of-8 M buckets, charge
   all padding, exact Q4_K affine correction, SwiGLU, weighting, cross-device
   synchronization, deterministic scatter, and queue drains. The complete
   median must be at most `8514.926 us`; the matrix portion must fit
   `5070.830 us`.
4. The NPU graph may be compiled offline with the installed driver compiler,
   but the runtime proof must load and execute its model-specific blob through
   the Level Zero NPU graph ABI. The executable may not link OpenVINO or
   oneDNN. GPU work remains repository-owned native code.
5. Stop on the first failed legality, all-value correctness, `96 GB/s`, or
   `8514.926 us` condition. Do not sweep compiler flags, bucket widths,
   partitions, precisions, graph APIs, or synchronization schemes.

## Consequences

- Full-model OpenVINO NPU compilation and NPU-only prefill are closed; their
  failures do not justify retrying the complete graph or tuning NPU alone.
- Seq653v2 authorizes one exact component implementation only. It does not
  reopen any GPU-only grouped family and does not promote an engine.
- The final engine still must pass the full teacher-forced, token, sentinel,
  smoothness, and 1k-256k performance matrix. No product speedup is claimed.
- If the exact component fails, ADR 0012's audit has no remaining independent
  architecture under the locked constraints and the next action is an owner
  contract decision, not another kernel variant.

## Evidence

- `output/product-architecture-feasibility-reconciliation-20260711Tseq653v2cleanZ/`
- `output/onednn-grouped-q4k-moe-component-gate-20260711Tseq650cleanZ/`
- `output/prefill-router-shape-census-gate-20260711Tseq639cleanZ/`
- `doc/adr/0003-surrogate-refine-splitplane-dual-phase-engine.md`
- `doc/adr/0012-close-grouped-prefill-select-product-feasibility-reconciliation.md`

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
