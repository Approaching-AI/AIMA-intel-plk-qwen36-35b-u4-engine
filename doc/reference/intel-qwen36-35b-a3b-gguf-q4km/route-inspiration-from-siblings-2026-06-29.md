# Route inspiration from sibling engines

Snapshot: 2026-06-29

Mined three sibling native-engine projects for reusable optimization directions.
Two are same-hardware + same-model; one is same-model on AMD (architecture only).
This is a candidate-direction ledger, not a commitment — `speedup_claims_allowed=false`.

## Sources and relevance

| project | hardware | model | relevance |
|---|---|---|---|
| `intel-box` | **Intel PTL / Arc B390** | **Qwen3.6-35B-A3B Q4_K_M (same sha)** | bandwidth-wall root cause; route-candidate matrix (37) |
| `intel-plk-highspeed` | **Intel PTL / Arc B390** | **same** | same-host llama.cpp baseline; Q4/Q6/MoE kernel corpus; DPAS reversal |
| `amd395-highspeed` / `-native` / `-0626` | AMD Strix Halo (arch only) | Qwen3.6-35B-A3B | serving loop, speculative verdicts, prefill route, correctness gate |

Not mined further (low transfer): `verge-*` (flux2 VAE / audio, not an LLM);
`pcie-prefill` / `a800-prefill` / `hwj-pd-prefill` (P/D-disaggregated multi-card
serving — out of scope for batch=1 single-host now).

## Cross-project converged findings

1. **The bandwidth wall is the GGUF K-quant in-place layout, not the hardware.**
   Triangulated three ways: intel-box measured raw GGUF source-stream peaks at
   Q6 67.8 / Q4 55.5 GB/s (= ~50% of LPDDR roof) while a clean packed_f32 layout
   hit 106 GB/s; intel-plk spent 17 days micro-tuning in-place dequant kernels
   and never closed its perf gate; amd395 in BF16 (no dequant, friendly layout)
   reached ~89% of hardware bandwidth. **Stop tuning in-place dequant kernels;
   change the layout.**

2. **The real same-host floor is higher than our bootstrap placeholder.**
   intel-plk measured llama.cpp **Vulkan** on the identical host+model:
   decode **~19.5 tok/s** (≈36.7 GB/s effective), prefill 246 tok/s @8k;
   OpenVINO prefill ≈2181 tok/s @8k. Our native decode (4.2 tok/s, 9–15 GB/s)
   **does not yet beat llama.cpp Vulkan.** Use 19.5 tok/s as the decode floor,
   not the 58 bootstrap number.

3. **DPAS reversal (correct the intel-box rejection).** intel-box rejected
   DPAS-i8 for hurting precision (rel_l2 0.003). intel-plk showed the right
   mapping — treat Q4 as small integers, integer dot, multiply scale *after* —
   is **bit-exact on Arc B390 (delta = 0.0)**. DPAS is still useless for
   *decode* matvec (memory-bound), but it is the **correct, precise tool for
   prefill GEMM** (compute-bound). The rejection was of one mapping, not of DPAS.

## Preferred directions (ranked)

### 1. Offline repack Q4_K/Q6_K into a streaming-friendly layout (highest ROI)
Root-cause fix, triangulated above. Repack each tensor so quant blocks +
scales/mins are contiguous **in the order the decode kernel streams them**, so
per-token dequant reads coalesce. Do it offline; never in the decode hot path —
this fixes scattered reads and dequant together. llama.cpp's own fast path
already repacks to `q4_K_8x8` + `ggml_gemv_q4_K_8x8_q8_K` (numerically
equivalent, ~1e-8) — reference it. Then re-run the source-stream probe on the
**repacked** layout to see if it beats 68 GB/s (neither Intel sibling did this).
intel-plk only tried half-packed (`packedq`, scales only) and it lost to scalar.

### 2. Persistent resident decode loop + streaming serving (also satisfies the streaming requirement)
batch=1 decode is launch/host-boundary bound. amd395's persistent full-layer
backend cut launch fanout **561 → 40 (14×)**; intel-plk's resident output head
went 4269 → 335 ms by moving the weight buffer lifetime out of the loop. This
also fixes our launch-bound small tensors (`attn_k`/`attn_v`/`*_shexp` at 2–8
GB/s). Build it as: resident weights/KV/SSM state + device-side LM-head top-1
fed back per token + per-step hook. **Streaming hook:** amd395's
`--defer-decode-token-cpu-sync` + per-step diagnostic hook is the natural SSE
attach point — change "read back once at loop end" into "read back every N
tokens and flush a chunk". (amd395 built the whole serving stack but explicitly
skipped SSE; the loop is reusable, add the chunked layer.) Intel equivalent:
Level-Zero/SYCL in-order queue + device reduction + buffer aliasing.

### 3. MoE routed-down is the real decode battleground (~92%)
intel-plk: decode time is ~92% in routed-MoE down, and it is kernel-body work
(not launch). Reusable fusion: `expert2pair` + `dualdot` (one dot, two experts)
+ `weightfold` (scale folded into the weight load) + gate/up↔down fusion +
activations in local memory. amd395's structural win: group per-expert work into
one matmul instead of per-token bmm. Source: intel-plk
`native/intel-plk-qwen36-native/src/gpu_opencl/matvec_microbench.cpp`.

### 4. Native prefill via DPAS/XMX GEMM (only physically winnable, unexplored frontier)
Neither Intel sibling wrote a native prefill kernel. Prefill is compute-bound
(weights reused across N tokens), DPAS compiles and is Q4-bit-exact, and the
roofline says a 3× prefill target does not violate the compute roof. Use vendor
SDPA + query chunking (amd395 ADR 0006: `q8192`/`block4096`), adaptive
short/long split (8k unchunked, ≥16k chunked), never materialize the full score
matrix, grouped-GQA streaming. OpenVINO prefill (~2181 @8k) is the bar.

### 5. Keep correctness on the teacher-forced distribution gate, not bit-exact
amd395's pivotal lesson: 50+ gates chasing 1-ULP bit-exactness was a dead end —
an FP64 sensitivity probe proved the target token sat at rank ~8000, so that
kernel's precision could never lift it. Gate on per-step logit ranking +
backend determinism (KLD<0.005, top-1≥0.99, boundary cosine≥0.999), not end
hashes, not length thresholds. We already do this — keep it. After bisecting to
a divergent boundary, run an FP64 sensitivity check before polishing it.

## Proven dead ends (do not re-run; cross-project)

- In-place dequant micro-variants (unroll/vectorize/pair/row): intel-plk 17 days, no gate close.
- Activation reuse (`xreuse`): intel-box 22 GB/s, slower than baseline.
- DPAS-i8 with activation quantized for **decode** matvec: hurts precision and memory-bound anyway.
- Half-packed (scales-only): intel-plk `packedq` never beat scalar — repack fully or not at all.
- Output-head-only optimization: intel-box — deleting lm_head entirely still misses decode by 8–21%.
- Speculative decode (MTP/DFlash): triangulated — amd395 MTP 87% accept but **net 42% slower** (single-token verifier over-budget); content diverges; only revisit if the loop can cheaply verify ≥2 positions.
- Single-op bit-exact death march: see direction 5.
- Heavy long-context profiling on the box: intel-box 32k profiling exhausted the 64GB host and dropped SSH — avoid.

## Streaming as a first-class constraint

The final engine must stream tokens (SSE/chunked), not return one batch JSON.
This is compatible with every direction above and is *served by* direction 2:
a persistent process owning the token loop, device-side per-token feedback, and
a deferred-CPU-sync hook flips to per-N-token flush. Our current batch-jsonl
verification mode is fine for the offline correctness phase, but R2/R3
architecture must evolve toward the resident streaming server. amd395's
`resident-http-server.py` is a reusable skeleton (OpenAI-compatible, tiered
readiness for clean TTFT/TPOT); it lacks only the SSE layer.

## Reusable artifacts (paths)

- intel-plk kernel corpus (Q4/Q6/MoE dequant+dot, sub_group reduce): `~/projects/intel-plk-highspeed/native/intel-plk-qwen36-native/src/gpu_opencl/matvec_microbench.cpp`
- intel-plk layout ADR + DPAS evidence: `~/projects/intel-plk-highspeed/doc/adr/0003-*`, `meta-log/2026-06-21.md` (P5-OD..OG), `meta-log/2026-06-11.md` (q4_K_8x8 repack)
- intel-box source-stream probe (port, then run on repacked layout): `~/projects/intel-box/native/intel-box-qwen36-engine/src/probes/ibx_source_stream_probe.cpp`
- intel-box byte-budget + roofline: `~/projects/intel-box/doc/active/.../*-full-layer-byte-budget-2026-06-15.md`, `*-roofline-2026-06-13.md`
- amd395 serving loop + contract: `~/projects/amd395-highspeed/tools/amd395-qwen36-35b-a3b-bf16-resident-http-server.py`, `~/projects/amd395-native/doc/cli-openai-contract.md`
- amd395 prefill route: `~/projects/amd395-highspeed/doc/adr/0006-q8192-block4096-accepted-long-context-route.md`
- amd395 correctness gate + bf16 reduction safe-points: `~/projects/amd395-qwen36-0626/doc/experiments/r3-selected-expert-moe-ledger.md`
