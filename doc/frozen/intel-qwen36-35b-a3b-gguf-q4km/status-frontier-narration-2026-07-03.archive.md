# STATUS + current-frontier narration archive (2026-07-03)

> Snapshot of the two Tier-3 state files immediately before the 2026-07-03
> doc-slimming (accepted-cut narration moved to the machine ledger
> `doc/active/intel-qwen36-35b-a3b-gguf-q4km/accepted-cuts.json`; budget/
> glide-slope machine state moved to `frontier.json`). Kept verbatim for
> the record; do not update.

---

## STATUS.md (as of 2026-07-02)

# STATUS - intel-qwen36-35b-a3b-gguf-q4km

> Single source of truth. Machine state -> `frontier.json`; closed -> `rejected-routes.json`; timeline -> `meta-log/`; first-read -> `current-frontier.md`.
>
> Snapshot: 2026-07-02

## NEXT ACTION
**The goal axis moved again; keep pushing resident/full-GPU decode-loop throughput.**
`frontier.json` now anchors the cold no-prefix diagnostic lane at `9.894275841`
tok/s from `output/r2-gpu-swiglu-read-drain-speed-20260703T030000Z/`, with
paired §1.5 distribution correctness from
`output/r2-gpu-swiglu-read-drain-distribution-20260703T031700Z/`
(`max_kld=0.004670076586`, top-1 rate `1`). The no-progress counter is `6`;
the row is below the 19.5 tok/s same-host Vulkan floor, so no speedup claim is allowed.

Ruler evidence:
- The recorded §1.5 distribution ruler originally failed on KLD while top-1
  stayed stable; CPU LM-head isolation reproduced the failure, so LM-head Q6 was
  not the cause. Details are in `meta-log/2026-07-02.md`.
- Resident linear state ownership now covers recurrent and convolution carry
  state in the shared runner, with convolution state using current/next
  ping-pong buffers instead of a per-layer copy-back.
- Accepted resident-loop cuts now cover selected/shared FFN scratch reuse,
  selected/direct expert-buffer matvecs, Q6/Q4 conv-state scratch reuse,
  post-conv/readback cuts, FFN-tail resident handoff, tail-output-to-RMSNorm
  resident input, router-from-resident-FFN-input, shared-gate host-value caching,
  Q4 CPU-order z read-as-drain, plain packed-Q4 read-as-drain, and SwiGLU
  handoff read-as-drain.
  Generic shared-Q6 down, attention-front residual/RMSNorm, router F32 scratch
  probes, and selected/expert8 SwiGLU->Q6 fusion are decode-closed; the direct
  layer-7 proof passes, but proof-backed decode regresses to `9.277108082`.
  The active FFN-tail path now reuses its input/intermediate/output CL buffers
  across calls while preserving input uploads, the existing single tail finish,
  and layer-output readback; this moved the current diagnostic best to
  `7.433283514` tok/s.
  Plain hidden RMSNorm scratch reuse and the already-proven linear
  attention-front handoff moved the best to `7.533730248` tok/s; extending the
  shared timed-loop model stream across the remaining hot helper opens moved it
  to `7.541657017` tok/s; later upload/readback cuts through selected cache
  misses, linear-delta, post-conv prep, readback cuts, full-core scratch, and
  plain resident-Q4 non-blocking Q8 uploads moved it to `8.075350694`; later
  resident-loop cuts moved it to `9.894275841`.
- Router logits now run through resident GPU F32 weights on the corrected
  baseline; serial-order F32 unroll is accepted, while WG256 reduction failed KLD.
- Selected-set concat no longer drains the OpenCL queue after every device-side
  copy; the consuming kernel observes the copies through the in-order queue.
- Resident cache diagnostics now reset hit/miss counters after untimed selected
  cache warmup. The measured loop is not a broad tensor-cache problem: the
  latest promoted row warms native-router top16 candidates and removes measured
  selected-cache misses (`0` gate/up, `0` selected-down, `0` Q6 down) while
  preserving top-1 and the distribution gate.
- The layer-5/6 bridge is closed on GPU: linear final RMSNorm uses a CPU-shaped
  final-norm kernel for the layer range 4..10. Device partial top-k cut the
  measured LM-head top-k readback/merge lane to roughly `160 us`, then the speed
  lane stopped auto-enabling the heavy live GPU router/attention/FFN trace
  capture. The new speed artifact is trace-free, top-1 exact, and carries the
  same generated IDs as the prior baseline. Reopening LM-head RMSNorm/Q8 device
  handoff with partial top-k regressed to `7.045365464` tok/s and is recorded
  closed in `rejected-routes.json`.
- Selected-down Q6 layout status: plane regressed `246.458` to `392.083 us`,
  pair2 regressed expert8 down `249.895` to `361.562 us`, and bpr2 regressed
  decode to `9.128097958`. Direct expert8 local64 is accepted only with the
  small-output raw-Q6 split; all-rows generic local64 is closed.
- Live selected-cache prewarm reached `6.955485469` tok/s, but its untimed
  prewarm cost `2070379382 ns`, so it is not a cold no-prefix row.

Next: push resident/full-GPU decode-loop speed. Router now consumes resident FFN
norm handles (`23118848 -> 20010726 ns`). Selected-only, grouped, and resident-only
FFN-input/no-readback routes are closed (`9.645536249`, `9.684578179` confirm,
`9.579176273`); reopen only with all-down-layout coverage or no SwiGLU host bridge.
Prewarmed shared-gate host caching, Q4 CPU-order z/plain packed-Q4/SwiGLU
read-as-drain are accepted; generic linear/LM-head host-row caches, selected
SwiGLU->Q6 tail-handle/no-readback fusion, selected expert-handle fastpathing,
resident packed-z replacement, and generic Q6/top-k read-as-drain are closed.
Isolated FFN-tail read-as-drain is also closed.
Large Q4 x8 is no longer the offline-repack question (`110.5220468`/`108.7925667` GB/s).
The L38/L39 FFN-residual value-sourcing class remains closed; exact-top-k over
free-run tokens is not the bar, and no speedup claim is allowed below 19.5 tok/s.

Context: backend = GPU on Arc B390 (user 2026-06-29). CPU engine is the
oracle/denominator (4.2 tok/s). Teacher-forced oracle + R0/R1/R2 carry over.
`speedup_claims_allowed=false`.

## Current Gate
| field | value |
|---|---|
| open gate | **GPU bring-up → R2 resident/full-GPU decode loop (Arc B390)** |
| machine state (Tier-2) | `frontier.json`: deepest layer 23, no-progress `6` since the latest goal improvement |
| goal anchor | best diagnostic decode `9.894275841` tok/s vs `19.5` Vulkan floor (`4.2` CPU denom); below floor, no speedup claim |
| attack / closed boards | `routes-ledger.json` (active: gpu_backend_bringup; parked: resident-loop, offline-repack, moe-down-fusion, dpas-prefill) / `rejected-routes.json` |
| gates | `frontier-sync.py` · `stall-gate.py` (hard = blocking) · `code-volume-check.py` (ratchet) · `validate_repo.py check_doc_discipline` |
| last closed denominator | `output/r2-floor-bind-20260629T052941Z/` |
| claim status | `speedup_claims_allowed=false` |

## Still Open
- [ ] **R2: resident/full-GPU decode loop** — move the goal metric from `9.894275841` toward the 19.5 floor (goal blocker)
- [x] replace the CPU/native layer-5 hidden-boundary bridge with a GPU-native layer-5 output / layer-6 input correction
- [x] recover decode-loop throughput after the final-norm correctness fix; current speed lane is `9.894275841` tok/s with paired distribution correctness
- [ ] collapse historical GPU probes/flags into the parameterized runner (`--layer`/`--z-source`) while keeping new work on the decode-loop path
- [ ] resident decode loop/API: token JSONL artifact smoke exists with count/id checks; still evolve off the batch driver
- [ ] native prefill via DPAS/XMX GEMM; 32k+ ladder only after a target-relative question
- [ ] promotion-grade benchmark matrix; speedup claims remain forbidden

## Closed Baseline
R0 target/model facts; R0 roofline and oracle bundle; resident harness load;
262144 denominator and 256k top-k unavailable lanes (`doc/adr/0001`, `0002`);
R1 native token replay; R2 fresh same-host denominator.
## Invariant: No speedup is promoted until §1.5 correctness stays closed and benchmark discipline is satisfied.

---

## current-frontier.md (as of 2026-07-02)

# Current Frontier — single pointer (Tier-3)

> Workstream: `intel-qwen36-35b-a3b-gguf-q4km`
> Snapshot: 2026-07-02

**New session reads this first, then `STATUS.md`.** This is the short Tier-3
pointer: state + where things live. It does **not** narrate per-boundary/per-layer
gates and does not paste per-token results — that is `output/` + `meta-log/`.
A harness gate (`check_doc_discipline`) fails this file if it grows into a lab
notebook (size, artifact-ref count, or any `N/8` result line).

## Where each thing lives

- **Machine state (Tier-2):** `frontier.json` — regenerate via
  `tools/intel-qwen36-frontier-sync.py` (do **not** hand-edit). Holds the goal-axis
  no-progress counter, the structural-freeze flag, and whether the current stall
  has a recorded review.
- **Attack board:** `routes-ledger.json` — active route, ranked parked alternates,
  candidate history, direction trigger, and `goal_stall_reviews` (keyed stall
  reviews that clear the hard gate).
- **Closed board:** `rejected-routes.json` — rejected routes + rejected classes.
  Do not re-run anything here without a new rationale.
- **Current gate + next action:** `STATUS.md`.
- **Timeline / story:** `meta-log/YYYY-MM-DD.md` (thin changelog).
- **Raw numbers (Tier-1):** `output/<dir>/` — the long dir name is the index; link
  it, do not retype it.

## State (read `frontier.json` for live numbers)

- **R0 closed:** target/model facts, roofline, oracle bundle, denominator.
- **Backend route (user 2026-06-29): GPU** on Arc B390. CPU q4-plane micro-tuning
  closed; the CPU engine is now the oracle/denominator (4.2 tok/s).
- **GPU bring-up open:** per-boundary teacher-forced gates reached layer 23; an R2
  GPU-hybrid decode smoke emits tokens with native-matching top-1. Best diagnostic
  decode is below the 19.5 same-host Vulkan floor; `speedup_claims_allowed=false`.
- **Ruler status:** the recorded §1.5 distribution check now passes on the
  resident loop after the GPU-native linear final RMSNorm correction. The fix was
  localized through a layer-5 CPU final-RMSNorm bridge, then moved into a shared
  OpenCL CPU-shaped final-norm kernel for layers 4..10 without CPU fallback. The
  convolution carry state now uses resident current/next ping-pong buffers, which
  removes the copy-back path, and selected live expert sets now reuse per-expert
  resident slices before device-side selected-set assembly. The selected gate/up
  SwiGLU path now also launches directly over eight resident expert buffers in
  router order, removing runtime selected-set Q4 materialization for that
  handoff. It now avoids the intermediate queue finish between the expert8
  matvec and SwiGLU kernels and reuses scratch CL buffers across calls. Selected
  Q6 down now uses the same direct resident expert-buffer pattern in the
  non-fused path, and its expert8 Q8/input and output CL buffers are reused
  without changing the explicit finish/readback semantics.
  The generic/shared Q4 gate-up -> SwiGLU handoff also reuses its Q8/input,
  source-map, gate-up, and SwiGLU CL buffers, and identity maps bypass the
  source-map/reorder kernel and enqueue matvec+SwiGLU with one drain.
  The linear-attention resident Q6 QKV + conv-state path now reuses transient
  Q8/input, QKV, conv-output, and fallback next-state CL buffers while preserving
  resident state ping-pong, explicit finishes, and readbacks.
  The sibling packed-Q4 resident conv-state path reuses the same transient buffer
  classes and non-blocking Q8 uploads. Resident linear-delta scratch uploads are
  now non-blocking too, selected expert cache-miss weight uploads are deferred
  during the measured loop, and post-conv prep now reuses raw/split/norm scratch
  CL buffers with a non-blocking raw input write. The resident Q6/Q4
  qkv+conv-state path can now skip QKV readback when trace and CPU-conv
  diagnostics are off. Generic shared-Q6, attention-front residual/RMSNorm, and
  router F32 scratch-buffer probes are closed in `rejected-routes.json`. Q4
  CPU-order z, plain packed-Q4, and SwiGLU handoffs now use read-as-drain in
  their resident paths.
  The active FFN-tail path now reuses its input/intermediate/output CL buffers across calls. Plain hidden
  RMSNorm now reuses its input/weight/output CL buffers while keeping per-call
  uploads and readback; resident norm weights now use prewarmed tensor handles
  in the timed loop. The already-proven
  linear attention-front handoff is now enabled on the current corrected command
  alongside full-core attention-front handoff.
  Router logits now use resident GPU F32 weights, and selected-set concat no
  longer forces a queue drain after every device-side copy. LM-head Q6 now uses a
  device partial top-k path on the speed lane. The speed lane no longer
  auto-enables the heavy live GPU router/attention/FFN trace capture for ordinary
  multi-token throughput rows. The generated decode runner now also reuses one
  model stream across the timed resident shell and remaining hot helper opens
  instead of reopening the GGUF in selected/shared/attention paths. The
  post-conv prep path can now skip diagnostic-only raw silu/q/k readbacks on the
  speed lane. The fused full-core attention-front handoff now reuses its local
  OpenCL buffers while preserving the same blocking writes, finish points, and
  residual/normalized readbacks. Plain resident Q4 matvec Q8 uploads are now
  non-blocking only on the opt-in `RunResidentPackedQ4X8` path, leaving closed
  handoff upload variants unchanged. Plain hidden RMSNorm now computes the
  scale in the original serial order, then applies it with a parallel kernel; a
  pure parallel reduction was faster but failed the distribution KLD gate and is
  closed. A faster WG256 router F32 reduction also failed the distribution KLD
  gate and is closed. Serial-order F32 matvec unroll16, FFN-tail fused output,
  resident linear-delta attention-readback skip, CPU shared-gate scalar,
  identity-map SwiGLU finish fusion, selected-Q6 read-as-drain, native-router
  selected-cache top16 warmup, resident norm-weight no-read handles, full-core
  resident-norm handoff, q/k norm host caching, linear-delta scratch reuse,
  down-to-tail handoff, tail-output-to-RMSNorm resident input, small-output
  raw-Q6 local64, expert8 selected-Q6 local64, router-from-resident FFN input,
  shared-gate host-value caching, Q4 CPU-order z read-as-drain, plain
  packed-Q4 read-as-drain, and SwiGLU handoff read-as-drain are accepted. Generic
  raw-Q6 local64 is accepted only for <=65536-row outputs, leaving the
  vocab-sized LM-head on the driver-selected local size. Diagnostic best is
  `9.894275841` tok/s with paired distribution correctness
  (`max_kld=0.004670076586`, top-1 rate `1`).

## Progress is judged on the goal anchor

The no-progress counter tracks the **goal metric** (cold no-prefix GPU decode
tok/s vs the 19.5 Vulkan floor), not the count of closed per-boundary gates
(methodology §1.1 / §3.3 failure-mode ④). As of 2026-07-02 the current counter is
`6` runs since the decode tok/s last improved, with the best still below the 19.5
floor.

- **Ruler:** the current best cold no-prefix speed row is
  `output/r2-gpu-swiglu-read-drain-speed-20260703T030000Z/`.
  It is top-1 exact and paired with
  `output/r2-gpu-swiglu-read-drain-distribution-20260703T031700Z/`,
  which passes the distribution KLD/top-1 threshold under teacher forcing.
  Boundary cosine is separate; full-logit cosine is diagnostic.
- **Route:** keep L38/L39 FFN-residual value-sourcing closed; continue the
  resident/full-GPU decode loop from the corrected linear final RMSNorm baseline,
  with speed as the current goal-axis target. Isolated selected gate/up
  resident-input Q8, grouped selected/shared FFN-input Q8, and resident-only
  FFN-input/no-normalized-readback are closed; reopen that boundary only with a
  plan that handles all down layouts or removes the SwiGLU host bridge.
  Linear/LM-head row caches, selected FFN fastpaths, resident packed-z
  replacement, and isolated Q6/top-k or FFN-tail read-as-drain are closed.

> Prior full narration: `doc/frozen/intel-qwen36-35b-a3b-gguf-q4km/`.
