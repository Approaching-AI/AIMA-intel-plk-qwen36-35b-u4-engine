# ADR 0001: R0 262144 Denominator Lane Unavailable

Date: 2026-06-26

## Status

Accepted for R0.

## Context

The acceptance matrix includes the 262144 input bucket, and R0 requires a
same-host denominator or a recorded interpretation before any product
performance work.

Two same-host denominator routes were attempted on `ptl-cls-dvt2-008` with the
locked GGUF model:

- OpenVINO GPU parsed the 262144 prompt, then failed during generation with
  `CL_OUT_OF_RESOURCES` even at `-mt 1`.
- llama.cpp/Vulkan passed preflight and 128/4096 smoke rows, but the 262144/1
  paired run timed out without producing a paired throughput metric. The
  lingering target process was terminated and cleanup evidence was recorded.

Evidence:

- `output/r0-openvino-denominator-20260626T041758Z/row.json`
- `output/r0-llama-denominator-preflight-20260626T045736Z/preflight.json`
- `output/r0-llama-denominator-20260626T050425Z/row.json`
- `output/r0-llama-denominator-20260626T050425Z/post-timeout-cleanup.json`
- `output/r0-denominator-oracle-boundary-resolution-20260626T070933Z/resolution.json`
- `output/r0-denominator-unavailable-policy-20260626T071453Z/policy.json`

## Decision

Accept the 262144 same-host denominator lane as unavailable for R0.

This closes the R0 denominator-lane interpretation item only. It does not
create a denominator metric, does not close R0, and does not authorize a
262144 speedup claim.

## Consequences

- No claim may say the native engine is faster than OpenVINO or llama.cpp at
  262144 until a valid 262144 denominator metric is captured or this ADR is
  superseded.
- 262144 rows may be reported only as absolute native measurements plus the
  denominator status `unavailable_by_r0_policy`.
- The raw OpenCL GGUF source-stream/qmatvec route remains rejected.
- Repeating OpenVINO GPU or same llama.cpp/Vulkan 262144 denominator attempts
  is out of scope unless the mechanism changes, such as different memory
  traffic, runtime, cache policy, model representation, or bounded benchmark
  contract.
- R0 still requires the full oracle bundle and resident harness
  `load(model, oracle_bundle)` evidence before it can close.

## Follow-Up

- Capture full-ladder teacher-forced distribution references.
- Capture per-boundary reference input and output tensors.
- Load the real oracle bundle through the resident harness.
- Supersede this ADR before making any future 262144 denominator speedup claim.
