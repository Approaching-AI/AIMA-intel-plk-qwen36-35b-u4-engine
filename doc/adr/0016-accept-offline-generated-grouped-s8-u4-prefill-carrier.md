# ADR 0016: Accept the offline-generated grouped S8-by-U4 prefill carrier

Date: 2026-07-11

## Status

Superseded by ADR 0021. The timing/code-generation evidence remains valid, but
seq671 shows that this carrier misses the locked component relative-L2 gate.

## Context

ADR 0015 reopened exactly one correctness-safe prefill dataflow at a fixed
`9771.436 us` real layer-27 cap. Earlier materialized-F16 composition failed
the SwiGLU boundary, while the F16 weighted-contribution plane and
deterministic F32 scatter were already proven correct. The remaining problem
was to retain Q8 input information, reconstruct the Q4_K affine-min term
inside the native accumulator, and avoid a promoted oneDNN dependency.

The selected implementation quantizes each token once to S8, gathers rows in
expert order, and executes grouped S8-by-U4 DPAS. It reconstructs each Q4_K
minimum term from compact six-bit minimum codes and signed low/high bytes of
the Q8 group sums, applies SwiGLU or normalized router weighting before the
F16 store, and scatters the eight contributions deterministically in F32.

Pinned oneDNN commit `01b479323f794da1a7a41a6fc084c7e11ccc2c3b` is patched
only to generate the two model-specific native OpenCL program binaries
offline. The measured runtime links only OpenCL and verifies that neither
oneDNN nor OpenVINO is mapped.

## Decision

Accept `full_tensor_grouped_s8_u4_compact_affine_f16_contribution` as the
prefill component carrier. Keep the checked-in generator patch, offline
prepack tool, standalone OpenCL runtime, and formal paired gate as the
reproduction path.

Clean seq663 supersedes seq662 as the engine-core reproduction. It retains the
dynamic router schedule input, moves the implementation behind a typed
`iq36_core` API and generated CMake target, and passes every terminal component
check:

- primary/confirm complete minimum, including schedule construction/upload,
  `9377.495 / 9422.735 us` versus `9771.436 us`;
- primary/confirm headroom `393.941 / 348.701 us`;
- paired median spread `0.174%`, inside the frontier noise discipline;
- all `4,194,304` SwiGLU, `16,777,216` weighted-down, and `2,097,152` routed
  values compared, with zero values above `5e-3`;
- both binaries report native `dpas.8x8` U4-by-S8, 256 GRF, SIMD16,
  workgroup `[32,4,1]`, and no spill/scratch declaration;
- the standalone runtime links and maps no oneDNN/OpenVINO library.
- one parameterized schedule implementation matches all 40 real 8k census
  layer shapes, from `155` to `222` active experts and through max group `994`.
- a smoke executable invokes the typed API directly rather than routing through
  the command-line parser.

## Consequences

- Both ADR 0015 component prerequisites are now closed: seq658 exact-Q6
  decode and seq660 grouped prefill.
- The next gate is a resident O(1)-in-layer-count loop that owns/reuses context,
  programs, prepacked weights, schedules, and scratch while binding both
  accepted carriers, followed by teacher-forced distribution and deterministic
  tokens. The current API still constructs one single-layer runtime per call.
- The native program and 541,065,216-byte resident selected-runtime payload
  are model-specific. Generic model loading remains out of scope.
- Offline oneDNN generation is allowed; runtime oneDNN/OpenVINO linkage,
  materialized-F16 residual variants, and datatype/workgroup sweeps remain
  closed.
- No token, context-ladder, or product speed claim follows from this component
  pass.

Evidence:

- `output/grouped-s8-u4-prefill-gate-20260711Tseq663engineapiZ/`
- `engine/gpu/opencl/onednn-grouped-s8-u4-fused.patch`
- `engine/include/intel_qwen36/grouped_s8_u4_prefill_runtime.hpp`
- `engine/src/grouped_s8_u4_prefill_runtime.cpp`
- `engine/tools/grouped_s8_u4_prefill_runtime.cpp`
- `tools/intel-qwen36-grouped-s8-u4-prefill-gate.py`

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
