# intel-qwen36 / doc Index

Snapshot: 2026-07-13

Stable conclusions, plans, references, SOPs, decisions, and ledgers belong in
`doc/`. Process history belongs in `meta-log/`. Current run state belongs in the
single `STATUS.md` board — not copied into multiple files.

## Where things live

| What | Where |
|---|---|
| **Current state / next action** | `doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md` |
| OpenVINO specialization roadmap | `doc/active/intel-qwen36-35b-a3b-gguf-q4km/intel-qwen36-35b-a3b-gguf-q4km-openvino-specialization-roadmap-2026-07-13.md` |
| Rejected/closed routes (machine-readable) | `doc/active/intel-qwen36-35b-a3b-gguf-q4km/rejected-routes.json` |
| Stable R0 bring-up plan | `doc/active/intel-qwen36-35b-a3b-gguf-q4km/intel-qwen36-35b-a3b-gguf-q4km-day0-r0-plan-2026-06-26.md` |
| Decisions (route rejections, policies) | `doc/adr/` |
| Archived append-only narration | `doc/frozen/intel-qwen36-35b-a3b-gguf-q4km/` |
| Session timeline | `meta-log/` |
| Workstream slug + route status | `doc/WORKSTREAMS.md` |
| Goal + acceptance shape | `goals/intel-qwen36-35b-a3b-q4km-engine.md` |
| SOPs (working discipline, R2, compare runner) | `doc/sop/intel-qwen36-35b-a3b-gguf-q4km/` |
| References (bandwidth-roofline reject) | `doc/reference/intel-qwen36-35b-a3b-gguf-q4km/` |

## Read order

The canonical new-session read order lives in **`AGENTS.md` → "Read order"**.
It is maintained in one place; do not duplicate it here.

## Mission

Build the fastest correct locked batch-size-1 OpenVINO GPU specialization for
the Qwen3.6-35B-A3B U4 IR on the Intel PTL target, against an untouched stock
OpenVINO ruler. Full mission and scope discipline: `AGENTS.md`.

## Current state

→ `doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`

The per-boundary compare narration that used to live in this file (≈400 lines of
append-only "Current State") is archived at
`doc/frozen/intel-qwen36-35b-a3b-gguf-q4km/doc-README.archive.md` and recorded in
`meta-log/`.
