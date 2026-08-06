# Agent Notes

This file is the canonical project guidance for coding agents. Tool-specific
files and other docs should point here instead of duplicating rules or reading
lists.

Spend time on thinking; you do not need to use the commentary channel to report progress to me.

Read and follow `.meta-agent/AGENT-RUNTIME.md` first, then the "Read order"
section below.

## Mission

Build the fastest correct OpenVINO-specialized inference engine for the locked
`Qwen3.6-35B-A3B` U4 IR on the local Intel PTL target machine.

The project is intentionally narrow:

- batch size is locked to `1`
- the product model and arithmetic semantics are locked to
  `/home/intel/Qwen3.6-35B-A3B-ov`; the prior GGUF model remains a diagnostic
  provenance/reference artifact
- dynamic batching, generic model loading, multi-tenant serving, and broad
  non-inference OpenAI platform API completeness are out of scope; ADR 0078
  adds the single-model resident OpenAI-compatible text-inference service
- untouched stock OpenVINO GPU is the correctness reference and performance
  denominator; OpenVINO GPU plus candidate-specific custom OpenCL operations
  are allowed final runtime dependencies
- stock and candidate workers must be isolated so candidate graph, custom-op,
  plugin-cache, or property changes cannot leak into the denominator
- no speedup is real without correctness, prompt/token evidence, and
  context-ladder smoothness evidence
- experiment drivers run directly on this machine; do not add a remote
  transport or require a separate target login

## Read order

Read these in order at the start of a new session. This is the single canonical
list; do not maintain a copy elsewhere.

1. `.meta-agent/AGENT-RUNTIME.md` — runtime / session SOP
2. `AGENTS.md` — this file (mission, discipline, paths)
3. **`doc/active/intel-qwen36-35b-a3b-gguf-q4km/current-frontier.md` — first-read Tier-3 pointer,
   then `STATUS.md` — current gate and next action**
   (machine state alongside them: `frontier.json` — Tier-2, regenerate via
   `tools/intel-qwen36-frontier-sync.py`, holds the noise-gated no-progress
   counters, glide-slope, and `goal_budget` kill-number;
   `accepted-cuts.json` — accepted cuts, do not re-litigate;
   `rejected-routes.json` — closed routes, do not re-run; `routes-ledger.json` —
   active route + parked alternates + direction trigger)
4. `doc/WORKSTREAMS.md` — workstream slug and route status
5. `goals/intel-qwen36-35b-a3b-q4km-engine.md` — goal and acceptance shape
6. `doc/active/intel-qwen36-35b-a3b-gguf-q4km/intel-qwen36-35b-a3b-gguf-q4km-openvino-specialization-roadmap-2026-07-13.md`
   — active specialization roadmap
7. `contracts/intel-qwen36-target-contract.json`,
   `contracts/qwen36-35b-a3b-openvino-u4-model-contract.json`, and
   `contracts/qwen36-35b-a3b-gguf-q4km-model-contract.json` — target, product,
   and legacy diagnostic facts
8. `benchmarks/intel-qwen36-35b-a3b-gguf-q4km/acceptance-matrix.json` — acceptance thresholds
9. `doc/active/intel-qwen36-35b-a3b-gguf-q4km/intel-qwen36-35b-a3b-gguf-q4km-day0-r0-plan-2026-06-26.md` — stable R0 plan
10. latest `meta-log/` entry — only for fresh changes

## Factory Discipline

- R0 comes first: target, oracle, roofline, and feasibility hard gate.
- Correctness uses the project's teacher-forced acceptance ladder.
- Implementation is O(1) in layer count: one parameterized layer plus a loop.
  Do not create per-layer source files.
- Explore in a hot resident loop; write full artifacts only for promotion.
  Exploration decode rounds run `decode-smoke --explore` (one JSONL line in
  `output/explore-log.jsonl`, no artifact dir); the full bundle is for
  promotion/confirm and correctness evidence only.
- Derive the kill-number before sweeping (ch.2 §2.1 #1): run
  `tools/iq36_budget.py` on the current best speed row. While the overhead-only
  ceiling is below the floor, overhead micro-cut candidates are sub-threshold
  by arithmetic — the next unit of work is kernel-side, or a recorded reason
  why not.
- Performance inference discipline: promotion is decided by a paired one-sided
  95% confidence bound, not a fixed repeat/confirm median-spread cutoff. Product
  rows use interleaved ABBA candidate/stock-OpenVINO blocks and require the
  lower bound to clear the ratio; latency components require the upper bound
  to clear the cap. Robust dispersion is environment telemetry only. Bundle micro-cuts that
  do not clear the inference gate; a new best still needs correctness evidence.
  See `doc/sop/intel-qwen36-35b-a3b-gguf-q4km/noise-and-explore-protocol.md`.
- When a route stalls, profile or change route. Do not sweep variants blindly.
  `stall-gate.py` enforces this: hard stall blocks; glide-slope + budget
  verdicts print the direction question before you launch the next probe.
- One state board: `STATUS.md` holds the current gate and next action. Do not
  copy current state into goals, plan, contracts, or READMEs. A component
  numeric match is not token correctness and is not a speed claim — that
  invariant follows from the open gate, so state the gate, not a per-artifact
  disclaimer. Per-run records belong to the machine layer (explore log,
  frontier census, ledgers); `meta-log/` gets per-session decisions and
  conclusions, not per-run narration.

## New session start

Follow the "Read order" above. The single source of truth for "where are we
now" is `STATUS.md`; progress narration belongs in `meta-log/`; archived
history in `doc/frozen/`.

## Path Discipline

The primary workstream slug is:

```text
intel-qwen36-35b-a3b-gguf-q4km
```

New files should follow these paths unless a document explains why not:

| File type | Path |
|---|---|
| status board (current state) | `doc/active/<workstream>/STATUS.md` — overwrite in place |
| session log | `meta-log/YYYY-MM-DD.md` |
| active doc | `doc/active/<workstream>/<workstream>-<topic>-YYYY-MM-DD.md` |
| frozen doc | `doc/frozen/<workstream>/...` |
| reference | `doc/reference/<workstream>/...` |
| SOP | `doc/sop/<workstream>/<topic>.md` |
| ADR | `doc/adr/NNNN-<topic>.md` (template: `doc/adr/TEMPLATE.md`) |
| handoff | `handoff/YYYY-MM-DD-<topic>.pending.md` |
| experiment output | `output/<experiment-name>-YYYYMMDD.../` |
| tool/script | `tools/<workstream>-<tool>.<ext>` |
| engine source | `engine/...` |

`handoff/`, `questions/`, and `answers/` are reserved for automation /
multi-agent flows and are unused in the current single-agent mode. Leave them
empty until such a flow needs them.

## Benchmark Discipline

Do not claim a speedup without recording:

- exact commit and diff scope
- host, kernel, firmware, driver/runtime versions, and model path
- prompt length, generated token count, precision, cache state, and warmup
- command used to run the benchmark
- raw output or a path under `output/`
- correctness sanity result
- accuracy level: component numeric, top-k/logit, deterministic token
  equivalence, or long-context sentinel
- smoothness result for any context-ladder claim
- for performance deltas: paired-block samples, the one-sided 95% confidence
  bound, and dispersion/environment telemetry; repeat/confirm median spread is
  diagnostic, not a hard pass/fail threshold

Cold no-prefix rows and prefix-hit rows are separate lanes. A prefill-only win
or decode-only win is diagnostic until both lanes clear the accepted target.

## Safety

- Treat firmware, kernel parameter, system package, and driver/runtime changes
  as high risk. Record the reason and rollback path before applying them.
- Do not commit credentials, raw tokens, private keys, or model-license
  material.
- Keep public-release hygiene from the start: use aliases and reproducible
  facts, and scrub private host details before changing repository visibility.
