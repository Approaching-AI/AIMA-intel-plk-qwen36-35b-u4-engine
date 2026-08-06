# ADR 0050: Close prefill-only GPU+NPU and restore owner decision

Date: 2026-07-12

## Status

Accepted. The project goal remains active pending an owner contract decision.

## Context

ADR 0049 admitted one fixed exact-Q4 NPU preflight before full hybrid wiring.
Clean seq767 consumes `73,728` real layer-27 gate/up rows at M64, representing
`19.327B` operations and slightly more than the fixed NPU third's `19.126B`.
The matrix window after the fused shell and scalar gate is `5506.039 us`; the
NPU must sustain `3.4736 TOPS`. The unchanged GPU source independently clears
its share at `8.7810 TOPS` versus `8.1174 TOPS` required.

The NPU source is correct and legal:

- its real-input slice passes cosine `0.999999751` and relative L2
  `0.000705773` over `32,768` outputs;
- the offline graph compiles to a `607,607,968`-byte native blob;
- fresh native Level Zero runs reproduce the compiler reference byte-for-byte
  and map no OpenVINO runtime.

Performance is terminal. Native repeat/confirm medians are
`16828.4 / 17108.7 us`, only `1.1485 / 1.1297 TOPS` versus `3.4736 TOPS`.
The paired spread is `1.666%` versus `0.5%`. The faster native row alone is
more than three times the NPU share deadline and more than twice the complete
`6250 us` FFN cap before any GPU work, cross-device synchronization, router,
scatter, shared expert, or final add is charged.

## Decision

Close `gpu_npu_prefill_only_exact_q4_complete_ffn_component_v1` at its
pre-registered rate and noise stop. Do not vary M, row count, expert grouping,
partition, graph topology, Q4 representation, datatype, precision, compiler
flags, runtime API, bucket, or synchronization. Do not build the full hybrid
FFN: zero GPU and zero helper cost cannot repair the measured NPU branch.

Restore ADR 0048's owner-contract next action. No evidence-backed native
prefill architecture remains under the simultaneous locked machine, GGUF
model, batch-1, component-correctness, native-runtime, and `1.10x` OpenVINO
contracts.

This is route exhaustion, not project completion, product acceptance, or a
speedup claim. Reopening requires an owner-recorded contract change or
independently verified new hardware/compiler capability with a complete bound
below `407.968 ms` before source implementation.

## Consequences

- The later GPU-only decode pass remains accepted for its scoped short lane;
  it does not compensate for absent product prefill.
- NPU decode, NPU-only prefill, and phase-split GPU+NPU prefill are all closed
  independently on measured source evidence.
- Another same-contract kernel would relitigate a terminal route rather than
  move the goal.

Evidence:

- `output/npu-exact-q4-m64-prefill-source-20260712Tseq767cleanZ/`
- `output/phase-split-gpu-npu-prefill-reopen-20260712Tseq766cleanZ/`
- `doc/adr/0049-reopen-prefill-only-gpu-npu-complete-ffn-gate.md`

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
