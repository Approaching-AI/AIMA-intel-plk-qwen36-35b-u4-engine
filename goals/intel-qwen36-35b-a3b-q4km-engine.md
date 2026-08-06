# Goal: intel-qwen36 OpenVINO-Specialized U4 Engine

Status: active
Created: 2026-06-26 · Product contract revised: 2026-07-14

## Objective

Create the fastest correct batch-1 inference engine for the locked
`/home/intel/Qwen3.6-35B-A3B-ov` U4 model on the local Intel PTL target. The
candidate is an OpenVINO GPU specialization and may use custom OpenCL GPU
operations, derived static graphs, and correctness-gated state precision or
layout changes.

The untouched stock OpenVINO GPU pipeline over the same locked model is both
the product correctness reference and the same-run performance denominator.
The candidate must deliver the material product win where context-scaling
cost dominates: at `32k/64k/128k`, both prefill and decode must independently
beat stock by at least `1.10x`. The exact `2k/4k/8k/16k` rows remain measured
non-inferiority guards. Winning only one long row, one phase, or a component
profile is not success.

Optimization work is selected from the long-context bottleneck first:
`32k/64k/128k` decode state traffic and context-dependent prefill are the
primary design axis. The `2k-16k` rows still have to pass correctness and both
phase guards, but they do not carry the same ten-percent optimization quota.
ADR 0075 records this distinction between priority wins and regression guards.

The long rows also determine the architecture. The promoted candidate may be
bucket-specialized: a long-context graph may use a custom bounded-state
attention carrier while a short-context graph retains stock SDPA, and gains in
another measured envelope may fund a short-row guard. No individual custom
component is required to beat stock in every short bucket. The binding result
is the complete bucket-selected candidate: priority rows clear `1.10x` in both
phases and guard rows remain within the explicit non-inferiority budget.

ADR 0070 is the owner-recorded contract change. The earlier GGUF/native engine
and llama.cpp oracle remain diagnostic evidence, not product semantics or a
final-runtime restriction.

## Runtime and Correctness Contract

- Product model: locked OpenVINO U4 IR and tokenizer bundle in
  `contracts/qwen36-35b-a3b-openvino-u4-model-contract.json`.
- Baseline: fresh isolated stock OpenVINO worker, no candidate custom-op config,
  no shared candidate plugin cache, no model mutation.
- Candidate: same OpenVINO release base and locked model, with every graph,
  custom-kernel, property, state-layout, and cache difference recorded.
- Correctness: candidate versus stock OpenVINO teacher-forced
  `KLD <= 0.005`, top-1 rate `>=0.99`, component cosine `>=0.999`, exact
  deterministic greedy tokens, and long-context sentinel truth.
- GGUF/OpenVINO disagreement is diagnostic. It cannot pass or fail a candidate
  after this contract change.

## Performance Target

The sole product reporting matrix is resident-model, batch 1, cold no-prefix
inference at exact input lengths `2k`, `4k`, `8k`, `16k`, `32k`, `64k`, and
`128k`, with exactly `512` output tokens. It has two performance roles:

- **Priority win (`32k/64k/128k`):** for each row and phase, the paired
  one-sided 95% lower confidence bound of candidate/stock throughput is at
  least `1.10x`, and the candidate also clears the absolute floor below.
- **Regression guard (`2k/4k/8k/16k`):** for each row and phase, the same lower
  confidence bound is at least `0.98x`. The two-percent margin is a maximum
  accepted product regression, not a fixed noise or stability test.

The stretch target is `1.125x` on all priority rows and `1.10x` on all guard
rows, independently in prefill and decode.

| priority input | candidate prefill minimum | candidate decode minimum | TPOT maximum | output512 total maximum |
|---:|---:|---:|---:|---:|
| 32k | 1768 tok/s | 37.16 tok/s | 26.91 ms | 32.312 s |
| 64k | 1317 tok/s | 29.28 tok/s | 34.15 ms | 67.248 s |
| 128k | 889 tok/s | 20.69 tok/s | 48.33 ms | 172.184 s |

The long-row absolute floors imply minimum end-to-end prefill cuts of
`58.027`, `77.947`, and `115.486 ms` per 1024-input equivalent at
`32k/64k/128k`, respectively, against the recorded stock anchors. The parked
OV1 adjacent-fusion route targets the measured `149.931 ms`
Transpose+GatedDeltaNet envelope, while OV2 attacks context-scaling attention
state traffic. A route must fund the matching priority-row cut under a complete
profile; a hidden-body or component cut alone is not a product result. The old
8k `40.896 ms` cut remains a stretch reference, not the primary admission
number.

Route selection and validation prioritize `32k/64k/128k`. The `2k-16k` rows
remain hard regression guards under their `0.98x` confidence-bound budget, and
all seven must pass their assigned role. Input `1k/100k/256k` and every
1024-output row remain diagnostic only.

Bucket specialization is part of one product candidate, not seven moving
targets. Each exact-bucket graph, state layout, and path decision must be fixed
before its paired stock/candidate blocks, recorded in the manifest, and used
consistently for all repeats. The stock worker remains untouched regardless of
which candidate path a bucket selects.

The exact input length is the logical prompt length, not a requirement to issue
one monolithic quadratic-memory inference call. A stock or candidate runtime
may use a fixed resident prefill-chunk schedule, provided every chunk stays in
the same request, positions and masks are continuous, and no state is imported
or reused from another request. Chunk size and count are frozen and recorded
per bucket before paired measurement. Prefill/TTFT covers the complete prompt
across all chunks; peak host/device memory and no-OOM cover the same lifetime.

The long-context decode target is intentionally hardware-reachable rather
than aspirational. Its required effective bandwidth is about
`96.416-99.885 GB/s`, below the measured `108.793-110.522 GB/s` packed-Q4
source bandwidth on this machine. A route still has to prove that bound with
prompt-conditioned OpenVINO semantics; zero-state or imported-state capacity
rows cannot satisfy it.

Raw prompts disable prefix caching and automatic chat templates. Runtime input
and output token counts must equal the requested row. Prompt lookup,
speculative decoding, prefix reuse, dynamic batching, and multi-stream
throughput cannot satisfy this lane.

## Roadmap

The stable route plan is
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/intel-qwen36-35b-a3b-gguf-q4km-openvino-specialization-roadmap-2026-07-13.md`:

1. OV0 freezes the stock ruler, captures the OpenVINO correctness bundle, and
   proves an isolated no-op custom GPU substitution.
2. OV1 establishes a bit-exact one-source GatedDeltaNet/layout replacement and
   profiles its real compiler boundary. Kernel micro-tuning stops when bounded
   candidates stall; prefill resumes only through an adjacent fusion with a
   complete bound that funds the matching `58.027-115.486 ms` priority-row cut.
3. OV2 is the long-context-first route. It keeps prompt state inside the same
   OpenVINO request, rejects cache-precision properties at the first
   correctness failure, and otherwise uses a bounded resident hot ring plus
   append-only compressed older state with fused attention/dequantization. It
   isolates arithmetic substitution from state ownership, then requires their
   integrated one-layer and all-layer semantics. The optimized graph must have
   one unambiguous request-state owner per full-attention layer; input-side
   mutation shared across fanned-out custom nodes is not a state contract. A
   correctness-only scalar carrier is not admitted directly to output512
   timing: a stock-shaped tiled implementation must preserve graph-owned state,
   mask/length metadata, and the bounded codec across resident prefill chunks
   while closing its decode dispatch/finalization bound. The route then crosses
   the 16k state transition and evaluates `32k/64k/128k` first. Exact-bucket
   specialization may keep stock SDPA in short rows when that is the faster
   full-model path; the short rows are subsequently run against the
   non-inferiority guard.
4. OV3 returns to the remaining prefill gap through measured
   DynamicQuantize/compressed-FC/shared-expert data movement, or the parked OV1
   adjacent fusion, only when a refreshed complete profile funds the cut.
5. OV4 compiles exact bucket variants and runs the full paired promotion
   matrix.

## Acceptance Shape

A promoted candidate must record:

- exact candidate commit/diff and locked IR/tokenizer digests;
- host, kernel, firmware-visible device, driver, OpenVINO Runtime/GenAI, and
  effective GPU plugin properties;
- isolated stock and candidate commands/config/cache paths;
- exact seven-bucket/512-output TTFT, prefill, TPOT, decode, total latency,
  peak memory, prefill chunk schedule/count, state precision/layout, kernel,
  and submit accounting;
- stock-referenced component, teacher-forced, exact-token, sentinel, and
  context-smoothness evidence;
- at least eight interleaved candidate/stock ABBA blocks per bucket and phase;
- a one-sided 95% lower confidence bound at or above `1.10x` independently for
  both phases of every priority row, and at or above `0.98x` for both phases of
  every regression-guard row;
- proof that no prefix-cache, chat-template, prompt-lookup, or speculative lane
  contaminated the result.

## Completion Rule

Completion requires one candidate that passes correctness, sentinel,
smoothness, memory/no-OOM, and the assigned performance role at all seven rows
in both phases. Each `32k/64k/128k` row must be a material `1.10x` win; each
shorter row must pass non-inferiority. Averages cannot hide a failing row.
Route rejection, a component win, a zero-state capacity row, or an 8k-only pass
does **not** complete this project goal.

The authoritative current gate and next action remain in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.

## Non-Goals

- multi-model serving or broad non-inference OpenAI platform API completeness;
- multi-model loading, dynamic batching, or batch size greater than 1;
- counting load/compile-cache improvements as resident inference speed;
- changing firmware, kernel, packages, or drivers without an explicit reason
  and rollback path.
