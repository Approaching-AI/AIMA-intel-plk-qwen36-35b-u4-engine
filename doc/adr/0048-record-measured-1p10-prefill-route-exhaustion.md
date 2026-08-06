# ADR 0048: Record measured 1.10 prefill route exhaustion

Date: 2026-07-12

## Status

Accepted pending an owner contract decision. The project goal remains active.

## Context

ADR 0047 required one evidence-only product reconciliation after the last
admitted complete-FFN source failed. Clean seq765 uses the locked 8k prefill
target, real component rows, same-host OpenVINO profile, and canonical route
decisions; it runs no target kernel.

The 1024-token product cap is `407.968 ms` (`1024 / 2510 tok/s`). An optimistic
replay charges only:

- seq764's faster matrix-only FFN row: `7.853124 ms * 40 = 314.124960 ms`;
- seq753's faster correct linear-state row: `2.166770 ms * 30 = 65.003101 ms`;
- seq758's faster projection-only row: `2.027162 ms * 30 = 60.814860 ms`.

Their sum is `439.942921 ms`, already `31.974793 ms` (`7.84%`) above the
complete tile cap. This is deliberately favorable: the FFN and projection
rows used for timing already fail component accuracy, and ten full-attention
layers, convolution/control, normalization/residuals, all non-matrix FFN work,
embedding, and final output are charged as zero.

The independent compiler reference does not create a hidden route. OpenVINO's
1024-token hidden-body profiled kernel sum is `411.882 ms`, `3.914 ms` above
the cap while removing the LM head and using converted U4 weights plus
synthetic hidden input. It is directional evidence, not locked-GGUF component
correctness.

The existing decisions already close the independent families: CPU, NPU and
GPU+NPU, grouped/handwritten/in-core exact-Q4 GPU, register/chunked/whole-linear
GPU, M8 and pinned F16/U4 complete FFN, and OpenVINO/oneDNN as final runtime
dependencies. Decode remains accepted only for the scoped short lane.

## Decision

Record that no evidence-backed native prefill architecture is admissible under
the simultaneous locked machine, GGUF model, batch-1, component-correctness,
native-runtime, and `1.10x` OpenVINO contracts. Close
`native_prefill_current_source_product_replay_v1` without implementing a
whole-tile replay: its optimistic core already exceeds the complete cap.

Do not launch another kernel, codec, layout, datatype, compiler-provider,
workgroup, or schedule variant under these unchanged contracts. The next
action is an owner decision changing one named dimension, or independently
verified new hardware/compiler capability with a complete source-derived
bound below `407.968 ms` before implementation.

This is route exhaustion, not goal completion, product acceptance, or a
speedup claim. It does not silently lower the `1.10x` target.

## Consequences

- Current native-prefill implementation work stops at a measured arithmetic
  boundary rather than relitigating closed variants.
- Reopening requires a recorded change to hardware/additional accelerator,
  model/precision, component accuracy, batch size, final runtime dependency,
  or minimum OpenVINO speedup ratio.
- Any same-hardware successor must first account for all omitted work and show
  a complete bound below the product cap; a component microbenchmark is not a
  reopen condition.

Evidence:

- `output/native-prefill-product-route-reconciliation-20260712Tseq765cleanZ/`
- `output/complete-ffn-microkernel-source-gate-20260712Tseq764cleanZ/`
- `output/openvino-hidden-prefill-profile-20260712Tseq751cleanZ/`
- `doc/adr/0047-close-f16-u4-complete-ffn-select-product-route-reconciliation.md`

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
