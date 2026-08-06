# Workstreams

Snapshot: 2026-07-23

This file is the source of truth for workstream naming and current route
status. Live gate/next-action state is in the STATUS board, not here.

## Naming Rule

Use `<repo-or-machine>-<model-family>-<precision-or-route>` when the repo has
multiple active lanes.

| Workstream slug | Hardware | Model / route | Status |
|---|---|---|---|
| `intel-qwen36-35b-a3b-gguf-q4km` | Intel PTL CLS DVT2, Core Ultra X7 358H | Qwen3.6-35B-A3B OpenVINO U4 specialization, batch size 1 | Active 2k-128k/output-512 candidate-vs-stock OpenVINO goal; clean 64k prefill/correctness carrier accepted, decode absolute floor and full promotion matrix remain open — see STATUS |

## Current Artifacts

| Purpose | Path / value |
|---|---|
| **status board (current state)** | `doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md` |
| target contract | `contracts/intel-qwen36-target-contract.json` |
| product model contract | `contracts/qwen36-35b-a3b-openvino-u4-model-contract.json` |
| legacy GGUF contract | `contracts/qwen36-35b-a3b-gguf-q4km-model-contract.json` |
| acceptance matrix | `benchmarks/intel-qwen36-35b-a3b-gguf-q4km/acceptance-matrix.json` |
| primary goal | `goals/intel-qwen36-35b-a3b-q4km-engine.md` |
| specialization roadmap | `doc/active/intel-qwen36-35b-a3b-gguf-q4km/intel-qwen36-35b-a3b-gguf-q4km-openvino-specialization-roadmap-2026-07-13.md` |
| R0 plan | `doc/active/intel-qwen36-35b-a3b-gguf-q4km/intel-qwen36-35b-a3b-gguf-q4km-day0-r0-plan-2026-06-26.md` |
| development methodology | historical project records; not a runtime dependency |
| workflow layer | `meta-agent` submodule |

## Route Guardrails

- Batch size `1` is a hard contract.
- The product target is the verified OpenVINO U4 IR bundle; the Q4_K_M GGUF
  artifact is diagnostic provenance and does not gate product correctness.
- The untouched stock OpenVINO worker is immutable. Candidate custom operations,
  graphs, properties, and caches must remain isolated from it.
- R0 must refresh target facts and roofline on the live Intel host.
- Promotion-grade manifests must bind target contract, model contract, and
  acceptance matrix by path and digest.
- Prefix-cache rows are separate and cannot count as cold-prefill progress.
