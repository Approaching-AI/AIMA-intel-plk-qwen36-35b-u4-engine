# OpenVINO specialization roadmap

Date: 2026-07-14

> This is the stable route plan created by ADR 0070. Current gate and next
> action remain in `STATUS.md`; experiments and route switches remain in the
> machine ledgers and `meta-log/`.

## Destination

Build a batch-1 PTL-specialized OpenVINO GPU runtime for the locked
`/home/intel/Qwen3.6-35B-A3B-ov` U4 model. The candidate may use custom OpenCL
GPU operations, derived static graphs, and state-layout/precision changes. The
untouched stock OpenVINO pipeline remains the immutable same-run correctness
reference and performance denominator.

Product completion still reports every exact `2k/4k/8k/16k/32k/64k/128k`
input row with exactly 512 output tokens, but ADR 0075 assigns two performance
roles. The `32k/64k/128k` priority rows must clear `1.10x` stock OpenVINO in
both prefill and decode. The shorter rows are hard non-inferiority guards: each
phase's paired 95% lower confidence bound must be at least `0.98x`. The
candidate may select a fixed graph per exact bucket: long rows may use the
bounded-state custom carrier while short rows retain stock SDPA. A custom
attention component need not be faster at a short shape; the complete
bucket-selected candidate must pass that row's assigned gate.

Exact bucket length means the complete logical prompt. Long prompts may be
processed as a frozen sequence of resident chunks in one InferRequest; this is
not prefix reuse or imported state. The manifest binds chunk size/count and
continuous positions/masks, while prefill/TTFT and peak memory cover the whole
sequence. A monolithic quadratic-memory debug call is not an acceptance
requirement.

The execution order is explicitly long-context first. OV1 established the
correct custom-operation/compiler boundary but its kernel-only micro-lane has
stalled; OV2 is active until a same-runtime prompt-conditioned state carrier
passes or its complete bound closes. This does not defer the long-row prefill
obligation: OV2's context-scaling cut, the parked OV1 adjacent fusion, and the
OV3 data-movement envelopes must together fund the matching priority-row cap
before promotion.

## Fixed arithmetic

| priority row | stock prefill | candidate floor | stock ms / 1024 | cap ms / 1024 | required cut / 1024 |
|---:|---:|---:|---:|---:|---:|
| 32k | 1607.0 tok/s | 1768 tok/s | 637.212 | 579.186 | 58.027 ms |
| 64k | 1197.0 tok/s | 1317 tok/s | 855.472 | 777.525 | 77.947 ms |
| 128k | 807.99 tok/s | 889 tok/s | 1267.342 | 1151.856 | 115.486 ms |

The measured Transpose + GatedDeltaNet envelope remains `149.931 ms` per
1024-input equivalent. The profile is component attribution only: it removes
the LM head, uses synthetic embeddings, and enables `PERF_COUNT`. A proposed
prefill route must be charged against the appropriate `58.027-115.486 ms`
priority-row end-to-end cut under a matching profile. The former 8k
`40.896 ms` cut is retained as a stretch/guard diagnostic, not the product
route kill-number. A component win does not count until end-to-end paired
evidence passes.

## Optimization thesis

The candidate is an OpenVINO specialization, not a wholesale OpenVINO rewrite.
Stock kernels remain the default wherever they already fit the bucket goal.
Work is admitted in this order:

1. **Remove context-scaling bytes.** For decode, keep one bounded request-owned
   hot window, compress older K/V once, and fuse dequantization with attention
   so old state is never expanded into a second full history. This is the main
   `32k/64k/128k` breakthrough route.
2. **Remove materialized prefill boundaries.** Fuse layout/Transpose with
   GatedDeltaNet, reuse DynamicQuantize across compressed FC consumers, and
   reduce shared-expert/MoE gather-scatter materialization. A custom kernel that
   merely reproduces one stock primitive without deleting a boundary is not a
   priority route.
3. **Specialize the compiled request.** Fix batch 1, exact bucket, state
   capacity, tile geometry, and scratch lifetime. Short buckets may select the
   stock path; long buckets may select the bounded-state path. Selection is
   frozen before paired measurement.
4. **Optimize launch/host overhead only from attribution.** At long context the
   dominant opportunity is GPU work and memory traffic. Dispatch, wrapper, or
   submit cuts are bundled unless a complete profile shows they can close a
   guard or a remaining priority-row gap.

Every route begins with a complete latency/traffic bound. If the bound cannot
fund the remaining row target, profile a different envelope instead of tuning
variants of the same kernel.

## OV0 — freeze the ruler

Purpose: make it impossible for candidate configuration to improve or alter
the denominator accidentally.

Required gates:

1. Verify every locked IR/tokenizer file against the product model contract.
2. Capture stock OpenVINO Runtime, GenAI, driver, firmware-visible device,
   effective GPU plugin properties, state tensor names/shapes/dtypes, and
   compiled-kernel labels.
3. Run stock and candidate workers in isolated processes and isolated cache /
   custom-op configuration paths. The stock worker loads no candidate config.
4. Capture a stock OpenVINO teacher-forced distribution/token bundle for the
   fixed short cases and the seven core prompt buckets.
5. Add a no-op candidate substitution mechanism and require exact token IDs,
   `KLD <= 0.005`, top-1 `>=0.99`, component cosine `>=0.999`, and no material
   paired performance regression.
6. Prove that a candidate custom GPU kernel can be selected in the compiled
   graph and that disabling its candidate config returns to the untouched
   stock implementation.

Exit: a clean, commit-bound baseline/oracle bundle plus a working no-op custom
GPU substitution. No optimization source is promoted before this exit.

Exit evidence: **closed** by
`output/openvino-specialization-bootstrap-20260713Tov0contractZ/`. The bundle
passes all locked-file/runtime/state checks, exact stock/candidate tokens and
all seven sentinels, 218 full-vocabulary comparisons at KLD `0`, and a
real-model custom GPU selection proof. Its 20-block paired mechanism overhead
has a one-sided 95% upper bound of `0.147084 ms/call`, below the registered
`2.148678 ms/token` 8k decode kill-number. Sequential full-model wall rows are
retained only as gross diagnostics and are not performance evidence.

## OV1 — GatedDeltaNet and layout fusion

Why first: the 1024-token profile puts `78.503 ms` in Transpose and `71.428 ms`
in GatedDeltaNet, and all 30 GatedDeltaNet nodes use the reference F16 kernel.
This is the largest directly replaceable prefill envelope.

Fixed design direction:

- keep a canonical state/activation layout across adjacent linear-attention
  operations;
- fuse the GatedDeltaNet gate, recurrent update, and output layout where the
  graph boundary allows it;
- eliminate pre/post Transpose materialization rather than merely making each
  Transpose faster;
- keep model state resident inside the OpenVINO `InferRequest`;
- use one parameterized kernel family, not per-layer source files.

Gates, in order:

1. Lock and replace one real layer at sequence 1024:
   - **1a boundary oracle:** non-invasively audit the actual F16 dispatch
     inputs, output, and final state. Do not add graph Results that perturb GPU
     precision or liveness.
   - **1b candidate substitution:** reproduce both captured outputs with one
     parameterized custom GPU operation, substitute it at the same real-model
     boundary, and pass component, final-logit, and all-state comparison before
     measuring performance.
2. Expand the same parameterized source without per-layer code:
   - **2a all-layer seq1024 numerics:** replace all 30 stock loops, prove that
     no stock GatedDeltaNet remains, and pass component, final-logit, and all 80
     state comparisons.
   - **2b performance-feasible carrier:** profile complete compiled stock and
     candidate graphs under one protocol. Attribute kernel time, materialized
     layout operations, reorders, and graph-fusion loss. Admit another source
     candidate only when its complete bound contributes enough to the matching
     `58.027-115.486 ms` priority-row cut after context-attention work is
     charged; no blind workgroup sweep and no cross-protocol arithmetic. If a
     source-equivalent custom
     kernel cannot approach the stock primitive, capture comparable compiler
     ISA, occupancy, register, and spill evidence before attempting fusion.
   - **2c teacher-forced length ladder:** once the carrier is at least
     performance-feasible, run the fixed-length component and all-layer
     distribution ladder before any end-to-end speed claim. This ordering does
     not relax the final correctness contract.
3. Combine the bundled Transpose+GatedDeltaNet cut with the measured OV2
   context-attention cut under one complete profile. Their lower-bound savings
   must fund the selected priority row; intermediate cuts are diagnostic and
   should be bundled.
4. The `32k/64k/128k` end-to-end prefill paired lower confidence bounds each
   reach `1.10x`.
5. Run `2k/4k/8k/16k` afterward as `0.98x` non-inferiority guards, with exact
   tokens, sentinel, memory, and smoothness checks.

Stop/switch rule: two bounded candidates that fail to move the registered
component or end-to-end confidence gate trigger a profile and route switch;
do not sweep workgroups, subgroup sizes, or precision blindly.

Gate-2b evidence through 2026-07-14:

- Seq806 captures the actual stock and custom IGC programs. Seq805 is not
  spilling; its dynamic SimpleGPU pitch arrays cause `75` indirect-stateless
  references, `160` GRFs / six EU threads, `1.9857x` instruction count, and
  `2.4973x` integer/address operations versus stock's `128` GRFs / eight
  threads. This explains the old `2.846x` component gap.
- Clean seq807 replaces those arrays with fixed-shape scalar indexing while
  preserving component outputs, final logits, and all 80 states bit-for-bit.
  Indirect stateless falls to zero, GRFs to `96`, and EU threads rise to ten.
  Custom GDN falls from seq805 `112.368 ms` to `49.170 ms` (`56.24%`), but
  same-run stock remains `39.287 ms`; this is a major accepted compiler cut,
  not a product win.
- The next two bounded candidates regress: disabling IGC's two-token time-loop
  unroll reaches `51.449 ms`, and explicit packed `half4` stores reach
  `50.858 ms`, versus the `49.170 ms` carrier. The stop/switch rule therefore
  parks GDN source micro-tuning. OV1 may resume only with a source-derived
  adjacent-fusion bound that contributes materially to a priority-row prefill
  cut and a matching paired end-to-end protocol; static ISA size or
  cross-protocol component arithmetic is not an admission argument.

## OV2 — same-runtime long-context state

Purpose: convert measured long-context decode bandwidth headroom into a
semantic product result without cross-runtime state import. This is the active
route after the OV1 source-micro stop.

Fixed feasibility evidence:

- The locked IR embeds `KV_CACHE_PRECISION=f16`.
- The existing block32 signed-INT8 K/V component, including current-token
  quantization and reduction, has a clean one-sided 95% latency UCB of
  `2.452708 ms/layer` at 128k versus its `2.825 ms` cap, with cosine
  `0.999999971822` and relative L2 `0.000324519358`.
- The fixed F32-hot8192/INT8-old-state zero-state diagnostic reaches
  `24.058 tok/s` at 128k versus the `20.69 tok/s` absolute floor. These facts
  establish capacity only; neither contains OpenVINO prompt-conditioned state.

Gates, in order:

1. **2a simple-property guard — closed/rejected.** Clean seq810 runs stock and
   two isolated `KV_CACHE_PRECISION=u8` workers at exact 2k/output20 after
   warmup. The two U8 outputs are deterministic with each other but both differ
   from stock's greedy text; diagnostic TPOT is also slower
   (`24.687/24.851 ms` versus `24.284 ms`). Stop before costly long rows. A
   property that compiles is not a correct state route.
2. **2b one-layer same-runtime attention carrier.** This gate has two ordered
   subgates:
   - **2b-i ABI and state ownership — closed.** Clean seq811 proves distinct
     dynamic multi-output SimpleGPU results (with the pinned plugin constraint
     `input_count >= output_count`). Clean seq812 locks the real layer-3 Q/K/V,
     mask, scale, append-axis, compiled SDPA, and same-request 2k prompt/decode
     ABI. Clean seq818 proves the bounded state topology: request-owned static
     F32 hot8192 K/V rings updated in place, append-only signed-INT8 older K/V,
     and logical FP16 symmetric block32 scales stored byte-exact in I8 Variable
     planes. The 16k-equivalent prompt/decode transition is bit-exact and has no
     full-history update dispatch.
   - **2b-ii-a arithmetic isolation — closed.** Clean seq819 replaces exactly
     layer 3's stock SDPA with one parameterized custom GQA operation while
     deliberately retaining the stock F32 append state. The seq812 decode
     component matches at cosine `0.999999991` and relL2 `0.000137512`;
     prefill/decode full-vocabulary KLD is
     `0.000150868/0.000065647`; stock and candidate greedy tokens are exactly
     `[271, 248068]`; all 80 state schemas remain finite and compatible. This
     accepts attention arithmetic and graph substitution only. The custom
     prefill profile is slower than stock and no timing in this gate is a
     performance claim.
   - **2b-ii-b bounded state integration — closed.** Clean seq822 at commit
     `8bf6bb3` replaces exactly layer 3 SDPA and its stock K/V Variables. Past
     and present shapes are derived from attention-mask/query shape, so later
     mask/shape consumers retain graph-owned O(1) metadata. The logical F32
     boundary is held as IEEE-F32 bits in I32 physical-guard8193 rings for a
     logical hot8192 window; the extra row prevents a one-token decode wrap
     race. Gate 2d later supersedes only this physical capacity with a full
     prefill-chunk guard while preserving the accepted logical semantics.
     Older K/V use signed block32 I8 plus byte-exact F16 scales, with one
     physical sentinel row carrying O(1) logical length. Exact 2k and
     8192-to-8194 runs execute one custom plus nine stock attention nodes and
     keep 84 finite states. All five teacher-forced rows pass at KLD
     `<=0.001927`; greedy paths, hot bits, cold length, I8 payloads, and F16
     scale bytes match stock/reference exactly. Online softmax failed the 8k
     first-decode gate at KLD `0.008563`, so the accepted carrier uses a
     two-pass reduction. No seq822 timing is a speed claim.
3. **2c all-ten semantic gate — closed.** Clean seq836 at commit `9545fcf`
   expands one parameterized source to layers
   `[3,7,11,15,19,23,27,31,35,39]`. It executes zero stock SDPA and ten custom
   operations, and owns 60 custom full-attention states alongside 60 untouched
   linear-attention states. One exact sink token is pinned ahead of the
   logical-hot8192 recent window and its historical one-token guard, making
   each seq836 physical hot state `[1,2,8194,256]`; older state remains signed
   block32 I8 with exact F16 scale bytes. Gate 2d supersedes the physical guard
   size for chunked prefill, not this semantic acceptance. All candidate-owned
   hot/cold transitions pass through cold length
   `0 -> 1 -> 2`. Exact 2k KLD is
   `0.000467902/0.000086406`; 8k-boundary KLD is
   `0.001075535/0.003292638/0.000002323`; greedy paths are exactly
   `[271,248068]` and `[271,248068,198]`. ADR 0074 accepts semantics only.
4. **2d single-state-owner tiled carrier — semantic/16k transition closed;
   exact-program performance active.** The rejected fanned-out split remains
   closed: it left query-visible hot K/V zero and failed decode KLD at
   `0.047932/0.535877`. The admitted topology is one custom node per layer with
   one request-state owner.

   Clean seq849 fixes the final composition error by matching stock
   `sdpa_micro`'s serial 16-lane denominator and cross-chunk accumulation
   order. Clean seq865 on the enlarged carrier reconfirms all-ten 2k/8k with
   five KLD rows
   `0.002267510/0.000485803/0.000163270/0.000437031/0.001009040`, exact greedy
   paths, finite untouched states, and exact candidate-owned codec transitions.

   Clean seq855/856 then identifies the rate blocker precisely. At 2k, the
   ten-layer candidate/stock attention envelope is `49.126/37.728 ms` prefill
   and `2.522/0.798 ms` decode. No program spills. The unified candidate
   prefill and decode shapes both compile the same `9249`-instruction,
   `192`-GRF, five-thread program, versus stock prefill at `3470`
   instructions/`256` GRF/four threads and stock decode at `2091`
   instructions/`160` GRF/six threads. The next cut is therefore static
   prefill/decode program separation, not a register, tile, or local-size sweep.

   A monolithic 16k direct-language-model debug call triggered global OOM on
   this 64-GiB host for both stock and candidate workers. This does not replace
   the existing product denominator; it closes only the monolithic correctness
   protocol. Commit `dd55eef` adds resident 8k chunks and expands the physical
   ring to one exact sink plus logical hot8192 plus an 8192-row write guard.
   Prefill reads the prior hot/cold state, writes the continuation into
   non-overlapping guard slots, and appends exactly the evicted rows to cold.
   Correctly rounded FP32 division makes every I8 payload and F16 scale byte
   deterministic at half-integer quantization ties.

   Clean seq864 crosses exact 16k at commit `dd55eef`: two resident 8k chunks
   plus the first teacher-forced decode pass with KLD
   `0.001589733/0.000308781`, exact greedy `[271,248068]`, cold length
   `8192 -> 8193` on all ten layers, and exact hot preservation and codec
   transitions. This is correctness evidence, not a speed claim.

   Remaining gate order:

   1. emit separate statically identified prefill and decode custom operation
      types/sources so each exact shape drops the other phase's code and state
      paths; recapture IGC and the complete attention profile;
   2. require the prompt-conditioned decode component UCB to stay below the
      registered `2.825 ms/layer` 128k cap, then measure complete chunked
      prefill without correctness-capture copies or swap;
   3. freeze the long-bucket chunk schedule and run `32k/64k/128k` output512.
      A miss triggers a new complete profile or the OV1/OV3 boundary-removal
      route, not arithmetic, block-read, chunk-size, or local-size sweeping.

   Exact 2k/4k/8k/16k candidate graphs may retain stock SDPA if custom decode
   is slower. Every complete short prefill and decode row must still clear the
   `0.98x` paired lower-confidence non-inferiority guard.
5. **2e priority length gate.** After gate 2d closes state semantics, the 16k
   transition, and the long decode component bound, run prompt-conditioned
   `32k/64k/128k` first, output512, with sentinel truth, memory growth, and
   paired one-sided 95% prefill and decode ratio lower bounds at or above
   `1.10x`. A component UCB, zero-state row, dirty prototype, or
   correctness-only carrier cannot substitute for this gate.
6. **2f guards and bucket composition.** After all three priority rows pass,
   freeze the per-bucket path and run `2k/4k/8k/16k`. Stock SDPA is allowed
   inside a short candidate graph, but untouched stock OpenVINO remains the
   isolated denominator and every full-model guard phase must have a paired
   95% throughput-ratio lower bound of at least `0.98x`. Any hot-window, sink,
   codec, or path-selection change requires a new accuracy and complete-bound
   gate, not a size sweep.

The dynamic SimpleGPU multi-output boundary is available, but generic dynamic
or trimmed `ScatterElementsUpdate` is not an accepted hot-state carrier: the
physical trace copies each full hot state during decode, and the expected
KVCache trim/update fusion is absent in the compiled pipeline. The accepted
topology instead mutates one request-owned F16x2-packed I32 K plane and one F16
V plane per layer, self-binds them after reset, and emits only current
cold-eviction scratch. Slot zero pins the exact attention-sink token. A
16384-row ring holds logical recent8192 plus a non-overlapping resident 8k
prefill write guard, giving 16385 physical V rows and a packed K shape of
`[1,2,1025,2048]`. Guard rows are stale capacity, not a second logical history.
Shape-derived lengths and the cold sentinel remain O(1) metadata and drive the
shared mask/shape graph without retaining removed full-history stock state.

## OV3 — secondary prefill envelopes

Use only if OV1 does not clear all prefill rows or a refreshed profile shows
these categories remain dominant.

Ranked routes:

1. Fuse/reuse DynamicQuantize with compressed FC; stock profile is `47.337 ms`.
2. Reduce repeated shared-expert quantization and compressed FC materialization.
3. Fuse MoE routing/gather/scatter around the already-efficient routed
   `MOE3Gemm`; optimize measured shared-expert FC and data movement, not the
   small fused node by assumption.

Each route needs a source-derived complete bound against the remaining
end-to-end gap before implementation. Micro-cuts smaller than that bound are
bundled and do not reset route-stall state.

## OV4 — exact bucket specialization and promotion

Once both phases have a winning kernel path:

- compile explicit batch-1 variants for the seven accepted context buckets;
- fix prefill tile, cache capacity/layout, and scratch allocation per bucket;
- preallocate `InferRequest`, state, and temporary buffers;
- tune single-request token scheduling rather than continuous batching;
- keep prefix caching, prompt lookup, speculative decode, and multi-stream
  throughput outside the cold-no-prefix product lane;
- use compiled-blob caching only for load/compile overhead, never as an
  inference-speed claim.

Promotion requires at least eight interleaved candidate/stock ABBA blocks per
bucket and phase, one-sided 95% lower confidence bounds at or above `1.10x` on
each priority phase and `0.98x` on each guard phase, the full
OpenVINO-referenced correctness ladder, long-context sentinel and smoothness,
bounded memory growth, and no OOM. No averaging can hide a failing bucket or
phase.

## Route order and parked alternatives

The active route is OV2's single-state-owner tiled carrier on the accepted
same-runtime hot/cold semantics. Its first gate is query-visible state and KLD,
then decode-overhead closure, the 16k transition, and the priority long product
rows. The route ledger parks, in order:

1. OV1 adjacent GatedDeltaNet/layout/projection fusion, with seq807 as the
   numeric/compiler carrier and no further kernel-only micro-tuning;
2. OV3 DynamicQuantize/compressed-FC/MoE fusion;
3. OV4 exact-bucket specialization.

Generic performance hints, extra streams, dynamic batching, prefix reuse,
speculative generation, the measured NPU split, cross-runtime state import,
and blind shape/precision sweeps are not fallback routes for this product lane.
