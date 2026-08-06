# Bandwidth-roofline reject table — batch=1 decode matvec

Snapshot: 2026-06-29
Tool: `tools/intel-qwen36-bandwidth-roofline-reject.py`
Artifact: `output/bandwidth-roofline-reject-20260628T161001Z/`

## Why this exists

Batch=1 decode reads every weight tensor once per token and runs an `M=1` GEMV
over it. Arithmetic intensity is about one MAC per loaded weight byte, so each
matvec lane is **memory-bound**: its floor is `weight_bytes / achievable_bandwidth`,
not the dot algorithm. This is methodology ch.2 trick #1 — derive the floor once
and let it reject a whole class of experiments, instead of sweeping compute-side
dot variants.

The post-R1 campaign on 2026-06-28 did the opposite: ~58 performance routes,
mostly compute-side dot variants (`direct` / `pair` / `pair-sum` / `block-sum` /
`row-pair` / `min-sum`, Q4 and Q6, on every lane), ~65–80% rejected, net accepted
gain about 2% of timed total. No route computed a bandwidth ceiling first. This
table is the missing one-page analysis.

## Method

```
effective bandwidth (GB/s) = weight_bytes / average_ns      # 1 byte/ns == 1 GB/s
weight_bytes               = (row_count/call_count) * (input_value_count/call_count) * bytes_per_weight
bytes_per_weight           = Q4_K 0.5625, Q6_K 0.8203125     # verified vs R0 source-stream probe
```

Two ceilings, both **R0-measured on the same host and model**:

| ceiling | Q4_K | Q6_K | meaning |
|---|--:|--:|---|
| R0 qmatvec achieved (`opencl_gb_s`) | 25.8 | 26.1 | an actual R0 qmatvec kernel already reached this |
| R0 source-stream (`source_gb_s`) | 48.5 | 64.2 | raw byte-stream readback, pure move, no compute |

Source artifacts: `output/r0-qmatvec-probe-20260626T043218Z/audit.json`,
`output/r0-source-stream-roof-20260626T042729Z/audit.json`,
profile `output/post-r1-resident-timed-20260628T054920Z/native-candidate-jsonl/native-candidate-stdout.json`.

## Table

Covered matvec time: `61,774,271,086 ns`. Bandwidth-only recoverable to the R0
qmatvec kernel ceiling: `35,177,525,459 ns` = **57% of covered matvec time**.

| lane | quant | calls | MB/call | eff GB/s | vs qmatvec | vs source | recover→qmatvec ns | verdict |
|---|---|--:|--:|--:|--:|--:|--:|---|
| `attn_qkv.weight` | Q4_K | 14100 | 9.44 | 11.4 | 44% | 23% | 7,934,439,159 | dot variants REJECTED |
| `ffn_gate_up_exps.weight` | Q4_K | 18800 | 9.44 | 14.6 | 56% | 30% | 5,309,532,915 | dot variants REJECTED |
| `attn_gate.weight` | Q4_K | 14100 | 4.72 | 9.1 | 35% | 18% | 4,729,751,753 | dot variants REJECTED |
| `ssm_out.weight` | Q4_K | 14100 | 4.72 | 9.4 | 36% | 19% | 4,494,993,626 | dot variants REJECTED |
| `output.weight` | Q6_K | 182 | 417.18 | 12.5 | 47% | 19% | 3,166,107,545 | dot variants REJECTED |
| `attn_q.weight` | Q4_K | 4700 | 9.44 | 10.9 | 42% | 22% | 2,353,051,052 | dot variants REJECTED |
| `ffn_gate_shexp.weight` | Q4_K | 18800 | 0.59 | 5.3 | 20% | 11% | 1,650,452,879 | dot variants REJECTED |
| `attn_output.weight` | Q4_K | 4700 | 4.72 | 10.6 | 41% | 21% | 1,235,618,765 | dot variants REJECTED |
| `ffn_up_shexp.weight` | Q4_K | 18800 | 0.59 | 6.7 | 26% | 13% | 1,213,908,647 | dot variants REJECTED |
| `ffn_down_shexp.weight` | Q4_K | 18800 | 0.59 | 8.0 | 31% | 16% | 1,170,607,277 | dot variants REJECTED |
| `attn_k.weight` | Q4_K | 4700 | 0.59 | 2.2 | 8% | 4% | 1,149,478,437 | dot variants REJECTED |
| `attn_v.weight` | Q4_K | 4700 | 0.59 | 3.9 | 15% | 8% | 769,583,405 | dot variants REJECTED |
| `ffn_gate_inp.weight` | F32 | 18800 | 2.10 | 34.4 | — | — | 0 | near ceiling, saturated |
| `ffn_gate_inp_shexp.weight` | F32 | 18800 | 0.01 | 3.2 | — | — | 0 | near ceiling, saturated |

## Conclusions

1. **Every dominant lane runs at 9–15 GB/s**, i.e. 35–56% of the R0 qmatvec
   kernel and 18–30% of the raw source-stream ceiling. The single largest lane,
   `attn_qkv`, runs at 11.4 GB/s — **2.3× slower than the simple R0 qmatvec
   probe kernel (26 GB/s)** on the same hardware and tensor. A day of `pair` /
   `direct` / `row-pair` variants on `attn_qkv` never caught up to the R0 probe's
   memory efficiency, because they optimize the wrong axis.

2. **The optimization budget is ~57% on the bandwidth axis, ~2% on the compute
   axis.** Closing the gap to the R0 qmatvec kernel ceiling alone recovers
   ~35.2e9 ns of matvec time; the entire 2026-06-28 dot-variant campaign moved
   timed total ~2% and mostly rejected. This is the textbook "optimize the lie"
   failure the ch.2 roofline check is meant to prevent.

3. **Reject the compute-side dot-variant class.** `direct` / `pair` / `pair-sum`
   / `block-sum` / `row-pair` / `min-sum` dot kernels do not move a memory-bound
   lane. Stop running them as separate candidates.

4. **Redirect to the memory-access path.** The real levers are weight layout for
   streaming, vectorized quantized loads, dequant pipelining / overlap, prefetch,
   and small-tensor launch overhead (`attn_k`/`attn_v`/`*_shexp` at 0.59 MB/call
   run at 2–8 GB/s — launch/sync bound, not compute bound). The only saturated
   lane is the F32 `ffn_gate_inp` router gate (34.4 GB/s) — leave it alone.

## Caveats (verify before promoting any speedup)

- The R0 ceilings are the **OpenCL device** path (`opencl_kernel=ibx_q*_k_qmatvec`).
  The post-R1 timed path uses CPU thread counts (`dense_matvec_threads=16` etc.).
  Confirm which backend the production matvec actually runs on, then compare
  against that backend's measured ceiling. The 26 GB/s GPU figure being 2.3×
  the production lane is itself a route question (is the engine on the slow
  backend?) worth a ch.3 review — but the memory-bound conclusion holds on either
  backend, since 9–15 GB/s is far below any plausible DRAM ceiling.
- `source_stream` is raw readback; a real dequant+GEMV will not reach it. Use the
  qmatvec-achieved column as the realistic, already-demonstrated target.
- This is a diagnostic floor analysis. A floor is not a speed claim;
  `speedup_claims_allowed=false` still holds until R2 + the benchmark discipline
  in `AGENTS.md` are satisfied.

## Reproduce

```bash
python3 tools/intel-qwen36-bandwidth-roofline-reject.py
```
