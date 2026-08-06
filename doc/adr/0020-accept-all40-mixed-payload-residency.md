# ADR 0020: Accept all-40 mixed payload residency

Date: 2026-07-11

## Status

Superseded by ADR 0021 for route progression. The capacity/load proof remains
valid, but the loaded Q4/Q6 carrier representations fail component accuracy.

## Context

ADR 0019 accepted the two codec carriers and the complete mixed schedule
budget, but used representative resident weights. The next gate required every
real layer-specific payload to be generated and loaded together before any
claim about a live 40-layer loop.

Clean seq670 reads the locked GGUF tensor index and generates:

- 20 Q4 layers at `541,065,216` resident bytes each;
- 20 Q6 layers at `645,922,816` resident bytes each;
- `23,739,760,640` total payload bytes.

Every expected file has the locked size and a recorded SHA-256. All 40 gate/up
weight hashes and all 40 down weight hashes are distinct. The native load smoke
creates one OpenCL context, loads four programs, returns 40 sequential layer
handles, and owns all `23,739,760,640` resident bytes. The process maps neither
oneDNN nor OpenVINO.

## Decision

Accept capture-free generation and one-context residency for all 40 real
mixed-codec layer payloads. Advance the route to execution across those handles
with each boundary consuming preceding live transformer state.

## Consequences

- Do not repeat layer-count, codec-census, payload-size/hash, or resident-capacity
  probes unless the GGUF or representation changes.
- The current proof performs no layer execution and therefore says nothing
  about live-state correctness or full-model performance.
- Teacher-forced distribution and deterministic tokens remain blocked until
  the live all-40-layer boundary loop passes component checks.
- The hard product target remains 1.10x same-run OpenVINO independently in
  prefill and decode at every accepted bucket.

Evidence:

- `output/all-layer-mixed-prepack-load-20260711Tseq670cleanZ/`
- `tools/intel-qwen36-all-layer-mixed-prepack-load-gate.py`
- `engine/tools/grouped_mixed_prefill_all_layer_load_smoke.cpp`

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
