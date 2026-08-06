# ADR 0017: Accept the resident dual-carrier boundary loop

Date: 2026-07-11

## Status

Superseded by ADR 0021. The resident ownership structure remains reusable, but
its grouped-prefill carrier is not component-accuracy compliant.

## Context

ADR 0015 admitted exact-Q6 decode and grouped S8-by-U4/F16-contribution
prefill only after their independent component gates passed. Seq663 made the
grouped path callable from `iq36_core`, but each call still reconstructed its
OpenCL state and loaded one layer's 541,065,216-byte payload.

Seq665 adds a resident grouped runtime. One object owns and reuses one OpenCL
context, three programs, static layer weights, dynamic schedule buffers, and
scratch. Its two-call smoke produces bit-identical routed output; the second
call is `9438.934 us` versus the `9771.436 us` cap. The full paired component
gate remains all-value correct at `9455.073 / 9405.233 us` primary/confirm.

Seq666 replaces the `layer.cpp`/`loop.cpp` skeleton with one parameterized
implementation and binds both accepted carriers on real model tensors:

- layer-27 grouped prefill: `9376.865 us`, zero of `2,097,152` routed values
  above `5e-3`;
- layer-7 exact Q6: `107.727143306 GB/s`, all `524,288` values compared to the
  CPU oracle, max abs `5.96046e-8`, zero mismatches;
- one context and one resident weight load per carrier in the process;
- no oneDNN/OpenVINO application linkage or mapping.

## Decision

Accept `NativeCarrierLayerRuntime` plus `NativeCarrierLoop` as the sole
O(1)-in-layer-count carrier-boundary integration. Continue from it; do not add
per-layer source implementations or return to standalone carrier variants.

This decision deliberately stops at the boundary loop. Seq666 supplies one
captured real layer per carrier and independent boundary inputs. It does not
yet load all 40 prepacked grouped layers, consume live preceding-layer state,
execute the complete attention/shared-expert/residual graph, produce tokens,
or establish a product speed row.

## Consequences

- Next generate/load the all-40-layer offline prepack and feed the carrier loop
  from live model state rather than capture files.
- Re-establish the locked teacher-forced distribution ladder, then
  deterministic token equivalence, before any context or product performance
  row.
- Runtime oneDNN/OpenVINO linkage, per-layer source duplication, NPU, raw-DP4A,
  and the old 1.25x board remain closed.

Evidence:

- `output/grouped-s8-u4-prefill-gate-20260711Tseq665residentcapZ/`
- `output/native-carrier-loop-gate-20260711Tseq666cleanZ/`
- `engine/include/intel_qwen36/native_carrier_loop.hpp`
- `engine/src/layer.cpp`
- `engine/src/loop.cpp`
- `tools/intel-qwen36-native-carrier-loop-gate.py`

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
