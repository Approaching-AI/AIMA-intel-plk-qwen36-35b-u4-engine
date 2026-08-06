# ADR 0077: Record OpenVINO locked-target route exhaustion

Date: 2026-07-15

## Status

Accepted pending an owner contract decision or independently verified new
capability. The project goal remains active and no acceptance threshold is
lowered.

## Context

ADRs 0070 and 0075 lock the product to the local PTL machine, the
`Qwen3.6-35B-A3B` U4 IR, batch 1, untouched stock OpenVINO GPU as the isolated
reference, exact seven-bucket output-512 coverage, `1.10x` dual-phase wins at
32k/64k/128k, and `0.98x` dual-phase guards at 2k/4k/8k/16k.

The accepted carrier remains correct but is not a product result. Clean
seq1171 -> seq1204 moves the exact 32k/output64 stable decode median
`31.977 -> 29.748 ms/token`; the absolute cap is `26.911 ms/token`, leaving a
`2.837 ms/token` complete cut. Seq1204 is sequential, output64, and diagnostic,
so it cannot substitute for the paired output-512 matrix.

The source-only seq1276 reflection audited the accepted carrier and every
registered closed-route reopen condition. The last two independently bounded
successors then closed:

- compiler-only seq1277 made the cold/hot partials spill-free at `96/64` GRF,
  but the sole seq1278 component regressed to median/UCB
  `0.641873/0.649374 ms` versus the `0.5618915-ms` cap;
- clean source-only seq1279 evaluated 100 real query/key boundary rows across
  stock/candidate, 2k/8k, every captured phase, and all ten full-attention
  layers. Group2/group4 clear the optimistic QK proxy but require `128/64`
  scale-separable K32 integer-DPAS calls versus 16 current K16 F16 calls.
  Every group with fewer calls (`32/64/128/256`) fails the same proxy; group32
  reaches worst QK relative L2 `0.002901504` versus `0.002` and is already
  closed by seq1255/1256 long deterministic correctness. No scale group is
  both numerically admissible and arithmetically cheaper.

The timing stop is independent of query-quantization implementation. Giving
query quantization, reduce, and update zero cost still leaves seq1275's partial
UCB at `0.576354 ms`, `0.0144625 ms` above the complete component cap. Charging
the unchanged measured reduce/update UCBs gives `0.587395 ms`. Seq1279 creates
no compiler, OpenCL context, GPU, or model-worker evidence.

The other complete axes remain closed by their registered evidence: group32
long correctness/performance, fine-codec single-owner and split topologies,
current context-attention prefill, OV3 DQ/FC/MoE prefill, OV1 adjacent prefill,
major-output/materialization, fixed FC, residual/elementwise, and dense-F16
attention. None has a reopen condition satisfied by current evidence.

## Decision

Close `openvino_integer_dpas_attention_arithmetic_v29c` and record that no
evidence-backed implementation route is admissible under the simultaneous
locked hardware, model/runtime, batch-1, correctness, memory/isolation,
seven-bucket output-512, and per-phase performance contracts on the current
software state.

Do not launch another source spelling, compiler gate, GPU/model worker, long
row, ABBA block, codec/group/tile/workgroup sweep, or favorable repeat under
unchanged contracts. A successor requires either:

1. an owner-recorded change to a named dimension such as hardware, OpenVINO or
   final-runtime capability, model/state precision, correctness tolerance,
   batch size, bucket/phase acceptance shape, or minimum performance ratio; or
2. an independently verified new compiler/hardware capability whose complete
   source-derived bound clears every applicable product cut before
   implementation.

This is infeasibility on the current recorded target/runtime evidence, not a
universal hardware theorem. It is route exhaustion, not project completion,
product acceptance, or a speedup claim.

## Consequences

- `STATUS.md` moves to an owner-contract decision gate.
- The accepted carrier and all reproducible component evidence remain useful,
  but none is promoted to an output-512 product candidate.
- The quantitative product target and correctness contract remain unchanged.
- Further experimentation is blocked until one of the two reopen conditions
  above is recorded.

## Follow-Up

- The owner must either keep the target recorded as infeasible on this state or
  change one named contract dimension.
- Supersede this ADR only when that decision or a newly bounded capability is
  recorded.

Evidence:

- `output/openvino-integer-dpas-attention-bound-20260715Tseq1279-cleanZ/`
- `output/openvino-hot-cold-partial-storage-specialized-component-20260715Tseq1278-cleanZ/`
- `output/openvino-route-exhaustion-reflection-20260715Tseq1276-cleanZ/`
- `output/openvino-direct-i8-product-20260715Tseq1256-all10-32k-o45-divergence-cleanZ/`

