# ADR 0018: Select grouped exact-Q6 prefill after the codec census

Date: 2026-07-11

## Status

Accepted as the next bounded route. No all-layer or product promotion is
implied.

## Context

ADR 0017 accepted a resident boundary loop using one captured real layer for
each previously accepted carrier. That proof did not establish that a single
grouped prefill representation covered every model layer.

Seq667 adds capture-free prepack generation and reruns the real layer-27 gate.
Primary/confirm complete time is `9319.659 / 9406.800 us` versus the fixed
`9771.436 us` cap, and every correctness/runtime check passes.

Seq668 then reads every real expert tensor in the locked GGUF:

- gate/up is Q4_K in all 40 layers;
- down is Q4_K in 20 layers and Q6_K in 20 layers;
- the 20 Q6_K down tensors total `4,404,019,200` raw bytes;
- real Q4 layers 5 and 27 can be generated without capture/oracle files;
- layer 27's eight payloads are byte-identical to the accepted payload;
- the two layer-specific `541,065,216`-byte payloads load in one resident
  native runtime.

The exact-Q6 carrier accepted by seq658 and bound by seq666 is an M=1
full-tensor decode shape. It does not implement the grouped M prefill down
operation, so it cannot fill the Q6 half of the model by substitution.

## Decision

Accept capture-free multilayer prepack for the Q4_K half. Set
`all_40_layer_prepack_ready=false` and select exactly one next route: a grouped
exact-Q6 prefill carrier for the 20 Q6_K down layers.

Before implementing that carrier, derive its kill-number from the complete
mixed-codec 40-layer budget and the measured router schedules. The passing
criterion is the complete layer sum under the locked 8k/1024 prefill target,
not a favorable single layer. Reuse the resident schedule, gather, gate/up,
SwiGLU, router-weighting, contribution, and scatter boundaries; do not add
per-layer source variants or application oneDNN/OpenVINO linkage.

## Consequences

- Seq667/668 close capture-free Q4 prepack generation and resident multilayer
  loading, but not all-layer execution.
- The Q6 route must prove all-value correctness and the mixed-layer timing
  budget before live transformer integration.
- After both codec families pass, extend the existing parameterized loop to
  live all-40-layer state, teacher-forced distribution, and deterministic
  tokens.
- The hard product target remains 1.10x same-run OpenVINO in both phases at
  every bucket. Component or structural evidence is not a product speed claim.

Evidence:

- `output/grouped-s8-u4-prefill-gate-20260711Tseq667prepackregressionZ/`
- `output/all-layer-prepack-feasibility-20260711Tseq668cleanZ/`
- `tools/intel-qwen36-all-layer-prepack-feasibility-gate.py`
- `engine/tools/grouped_s8_u4_prefill_multilayer_load_smoke.cpp`

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
