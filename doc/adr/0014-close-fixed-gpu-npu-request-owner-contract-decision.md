# ADR 0014: Close the fixed GPU+NPU route and request an owner decision

Date: 2026-07-11

## Status

Superseded by ADR 0015. Its closure of the fixed GPU+NPU route remains valid.

## Context

ADR 0013 admitted exactly one `2:1` GPU:NPU component and required it to stop
at the first legality, correctness, `96 GB/s`, or complete-prefill failure.
It prohibited partition, representation, compiler, precision, bucket, and
synchronization sweeps.

Clean seq654 first proves the runtime boundary is legal.  A model-specific NPU
blob is compiled through the installed driver compiler, extracted, and loaded
in a fresh process through the Level Zero graph ABI.  The application links
`libze_loader` but not OpenVINO or oneDNN, maps neither library, and reproduces
all `64 / 64` reference bytes.  The system NPU driver maps its own compiler
library internally even for a native blob; that is driver behavior rather than
an application dependency.

Clean seq655 then executes the fixed M=1 NPU third of the real
`output.weight` Q6_K tensor:

- GPU owns `165,568` rows and NPU owns `82,752` rows; the NPU consumes
  `139,023,360` raw Q6_K bytes and one real Q8_K final-normalized vector.
- The source graph preserves every low-4 code, high-2 code, block scale, and
  signed group-16 scale.  Its representation matches the native Q6_K/Q8_K
  oracle at relative L2 `1.31595e-7`.
- All `82,752` NPU outputs pass the component ruler: cosine
  `0.9999998366`, relative L2 `0.000571892`, and max abs `0.0137329`.
- The `169,476,120`-byte source IR becomes a `683,537,744`-byte native blob,
  `4.9167x` the raw Q6_K payload.  A fresh native process reproduces all
  `331,008` output bytes and maps no OpenVINO library.
- Best native execution is `10,594.7 us`; median is `13,210.4 us`.  That is
  only `13.122 / 10.524 GB/s` raw-Q6-equivalent, versus the NPU share's
  required `31.992 GB/s`.

The terminal bound is independent of the GPU implementation: charge the GPU
zero time and divide the complete `417,177,600` Q6_K bytes by the best NPU
time.  The resulting paired ceiling is only `39.376 GB/s`, versus the locked
`96 GB/s` kill-number.  It misses by `2.438x`.  Host-runtime removal,
concurrency, and a faster GPU cannot repair that bound.

## Decision

Close `gpu_npu_parameterized_exact_q6_variable_m_component_v1` on the fixed
decode bandwidth gate.  Do not run the 26-bucket variable-M prefill component:
ADR 0013 says to stop at the first terminal failure, and decode has already
failed.

No implementation route remains authorized under the simultaneous locked
constraints: this PTL machine, this GGUF Q4_K_M model, batch size 1, the current
correctness contract, no OpenVINO/oneDNN final runtime dependency, and the
`1.25x` per-phase OpenVINO product target.  The next action is an owner contract
decision, not another kernel or compiler variant.

This decision does not lower the product target, change the goal, declare the
project complete, or claim a native speedup.

## Consequences

- Exact low4/high2 NPU carrier variants, partition ratios, compiler flags,
  graph shapes, precision hints, and native/OpenVINO dispatch comparisons are
  closed on this software and hardware state.
- The seq653v2 proxy remains useful architecture screening evidence, but it is
  superseded by seq655's source-exact native measurement.
- Existing exact-refine traffic cuts remain preserved components; they do not
  form a passing product architecture by themselves.
- Reopening requires an explicit owner-approved change to at least one locked
  contract dimension, or independently verified new NPU compiler/hardware
  capability that preserves compact Q6_K semantics and first derives at least
  `31.992 GB/s` for the fixed NPU share without a tuning sweep.

## Follow-Up

The owner must choose whether to keep the target and record infeasibility, or
change a named contract dimension such as hardware, model/precision,
correctness, final runtime dependencies, batch size, or performance ratio.
Until that choice is recorded, no further experiment is authorized.

Evidence:

- `output/npu-level-zero-blob-legality-20260711Tseq654cleanZ/`
- `output/npu-exact-q6-representation-20260711Tseq655cleanZ/`
- `doc/adr/0013-select-gpu-npu-parameterized-exact-component-gate.md`

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
