# GPU Backend Bring-Up (thin)

Workstream: `intel-qwen36-35b-a3b-gguf-q4km`
Status: active route. Not a promoted speedup claim. `speedup_claims_allowed=false`.

> This is the thin active board. **Live machine state:** `frontier.json`
> (regenerate via `tools/intel-qwen36-frontier-sync.py`). **Per-gate blow-by-blow
> (31 handoff gates, layers 5–23):** `doc/frozen/intel-qwen36-35b-a3b-gguf-q4km/gpu-backend-bringup-2026-06-30.narration.archive.md`.
> **Raw artifacts:** `output/gpu-*` (the long dir name is the index — link, do not
> retype). Do not re-grow this file into a lab notebook; the validator caps
> `doc/active/*.md`.

## Route

Backend route selected by user 2026-06-29: GPU native on Arc B390. CPU q4-plane
micro-tuning closed; the CPU engine is now the oracle/denominator, not the
target. Start from the same-hardware intel-plk OpenCL kernel corpus (Q4/Q6/MoE
dequant+dot) + offline repack layout.

**First goal:** reach/beat the same-host llama.cpp Vulkan decode floor
`19.5 tok/s` (CPU native is `4.2`), under the existing teacher-forced oracle.

## Current gate (see frontier.json for live numbers)

- **Structural axis:** per-boundary teacher-forced GPU gates are closed
  layer-by-layer through **layer 23** (deepest closed). Every gate is a captured
  single-token handoff and explicitly **does not prove decode/token/throughput**.
- **Goal axis:** **no GPU decode-lane run exists yet** — `current_best_tps = null`
  vs the `19.5` floor. `frontier.json` reports `runs_since_goal_improved` (149 GPU
  probe runs, goal never moved) → **soft reflection breached**.

> **D-controller reflection (ch.3 §3.5, recorded 2026-06-30):** the structural
> axis advancing while the goal axis stays flat is the highspeed trigger-① shape
> (local progress on a route that is dead *for the goal*). The correction is
> **within-route**: stop adding per-boundary gates and **assemble the R2 decode
> loop** so the goal metric exists. The engine layer is already O(1) — run one
> parameterized GPU layer + the loop; do not re-verify every layer index. Run
> `tools/intel-qwen36-stall-gate.py` before launching another probe. GPU itself
> is not in question, so no route switch — see `routes-ledger.json#parked_routes`
> only if the goal proves unreachable.

## Next implementation gate

Two lanes, in order:

1. **Close the last structural gap (small):** layer-23 FFN/l_out from live GPU
   `l_out-22`, keeping the backend Q4 CPU-order layer-22 z correction active
   (`iq36::RunQ4KCpuOrderMatvec` before `ssm_out.weight`). Layer 23 is in
   `FULL_ATTENTION_LAYERS` — do not route it through the linear conv-history
   path; use all-history KV; verify V tensor type / Q4-Q6 packed layout in the
   probe. **Fold this into the parameterized GPU runner (`--layer 23`), not a new
   per-layer probe file** — the code-volume ratchet now fails a new `*-probe.py`.
2. **Cross R1→R2 (the actual goal blocker):** assemble one parameterized GPU
   layer + decode loop into a running engine that emits a token, then a first
   cold no-prefix decode tok/s vs the `19.5` floor. Keep CPU/native as the
   correctness denominator; no throughput claim before prompt/token evidence,
   top-k/logit evidence, and ladder-smoothness evidence.

Do not port the whole microbench monolith. Keep the resident
load-once/run-many boundary; do not regress to process-per-probe.

## Decision (2026-06-30)

Proceed with GPU bring-up from the intel-plk OpenCL corpus. The runtime gate,
raw/repacked stream gates, Q4 x8 packed qmatvec, and the full per-boundary chain
(QKV → preconv/conv/postconv → delta recurrent → output projection → router →
selected/shared FFN → aggregation → residual → layer shell) are closed and
composed into resident single-layer shells through **layer 23** (full list +
metrics: frozen narration archive). The next gate is **layer-23 FFN/l_out then
the R2 decode loop**, not a full backend port and not more per-layer gates.
