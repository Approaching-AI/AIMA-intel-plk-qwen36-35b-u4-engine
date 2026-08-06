# Long-context performance target

Snapshot: 2026-07-13 · Acceptance roles revised: 2026-07-14

## Decision

The sole product performance matrix is batch-1, cold-no-prefix inference with
exact input lengths `2k`, `4k`, `8k`, `16k`, `32k`, `64k`, and `128k`, and
exactly `512` generated tokens. ADR 0051 removes `1k`, `100k`, `256k`, and all
`1024`-output rows from promotion; their existing artifacts remain diagnostic.

ADR 0075 assigns two roles without removing any row. At
`32k/64k/128k`, the candidate OpenVINO specialization must beat an isolated,
untouched same-run stock OpenVINO GPU U4 worker by at least `1.10x`
independently in prefill and decode; the effective threshold is
`max(absolute target below, 1.10 * same-run stock median)`. At
`2k/4k/8k/16k`, each phase is a non-inferiority guard whose paired one-sided
95% candidate/stock throughput-ratio lower bound must be at least `0.98x`.
The two-percent margin is a product regression budget, not a noise estimate.
No phase, bucket, or aggregate-score averaging is allowed.

## Core scorecard

The TTFT and total-latency caps below are hard floors for priority rows and
stretch references for guard rows. A raised same-run priority threshold
produces correspondingly tighter caps.

| input / output | role | locked stock OpenVINO prefill / decode | 1.10x prefill reference | 1.10x decode reference | TTFT reference | TPOT reference | total reference |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2k / 512 | `0.98x` guard | 2206.401 / 44.226 tok/s | 2428 tok/s | 48.65 tok/s | 0.843 s | 20.55 ms | 11.368 s |
| 4k / 512 | `0.98x` guard | 2347.302 / 44.024 tok/s | 2583 tok/s | 48.43 tok/s | 1.586 s | 20.65 ms | 12.158 s |
| 8k / 512 | `0.98x` guard | 2281.314 / 42.359 tok/s | 2510 tok/s | 46.60 tok/s | 3.264 s | 21.46 ms | 14.251 s |
| 16k / 512 | `0.98x` guard | 1987.589 / 39.033 tok/s | 2187 tok/s | 42.94 tok/s | 7.492 s | 23.29 ms | 19.415 s |
| 32k / 512 | `1.10x` priority | 1607.000 / 33.773 tok/s | 1768 tok/s | 37.16 tok/s | 18.534 s | 26.91 ms | 32.312 s |
| 64k / 512 | `1.10x` priority | 1197.000 / 26.617 tok/s | 1317 tok/s | 29.28 tok/s | 49.762 s | 34.15 ms | 67.248 s |
| 128k / 512 | `1.10x` priority | 807.990 / 18.806 tok/s | 889 tok/s | 20.69 tok/s | 147.438 s | 48.33 ms | 172.184 s |

Route selection and profiling should prioritize the `32k`, `64k`, and `128k`
bottlenecks. The `2k` through `16k` rows remain hard regression guardrails, not
an alternate short-context headline.

## Runtime and denominator state

The OpenVINO model is `/home/intel/Qwen3.6-35B-A3B-ov`, with U4-compressed
constants. ADR 0070 makes this locked IR the product model and permits OpenVINO
GPU plus candidate-specific custom OpenCL operations as the final runtime. The
untouched stock worker is both the correctness reference and performance
denominator; candidate graph, property, custom-op, and plugin-cache state must
be isolated from it.

The locked values above are the maximum, per phase, of the audited same-host
rows and the corrected raw-prompt medians. Clean seq769 completes the
missing output-512 `prefill_shape` and `sentinel` rows at `2k`, `4k`, `16k`,
`32k`, `64k`, and `128k`: all `36/36` measured rows have exact input/output
counts with prefix caching and chat templates disabled. The full three-prompt
core denominator is now complete. It raises only the 16k decode floor from
`42.76` to `42.94 tok/s` and the 64k decode floor from `29.19` to
`29.28 tok/s`; no prefill floor rises.

The raw-prompt protocol disables both prefix caching and automatic chat
templates, ignores EOS only to enforce the fixed 512-token request, and checks
that the runtime-reported input and output token counts match the row. Existing
Evidence is in
`output/r0-openvino-denominator-matrix-20260713Tseq769-raw-core-rest-cleanZ/`;
the original derivation remains in `performance-target-2026-07-11.md` and the
acceptance matrix.

## Hardware meaning

The strict decode inventory is 1.975676544 GB of active weights and output head
per token plus `context * 20,480` bytes of FP16/BF16 KV reads. The seven targets
therefore require approximately `96.416-99.885 GB/s`. Real packed-Q4 carriers
have measured `108.793-110.522 GB/s`, so the long-context decode target remains
above noise yet below measured source-bandwidth capability.

The old native prefill families remain closed, but they no longer constrain the
final runtime. The priority floors require end-to-end cuts of `58.027`,
`77.947`, and `115.486 ms` per 1024-input equivalent at `32k/64k/128k`.
The stock hidden-body profile puts `78.503 ms` in Transpose and `71.428 ms` in
GatedDeltaNet, so their adjacent fusion is one useful envelope; bounded-state
attention must contribute the context-scaling cut. The old 8k `40.896 ms`
number is a stretch diagnostic. Component profiles cannot substitute for the
paired end-to-end gate.

## Promotion protocol

- batch size 1, resident model, cold prompt and prefix-cache state;
- three exact prompt classes per input bucket: `prefill_shape`, `sentinel`, and
  deterministic `filler`;
- warmup outside measurement, then at least eight interleaved ABBA paired
  candidate/stock-OpenVINO blocks per bucket and phase;
- stock and candidate rows on the same host, software stack, and power state,
  but in isolated workers and candidate configuration/cache paths;
- median prefill/decode per prompt and phase, raw TTFT/TPOT distributions, and
  a one-sided 95% lower confidence bound of the paired candidate/stock ratio
  that clears the row's required ratio (`1.10` priority or `0.98` guard);
- component latency uses at least twenty samples and must keep its one-sided
  95% upper confidence bound below the registered cap; repeat/confirm spread
  and robust CV are environment diagnostics, not performance vetoes;
- stock-OpenVINO-referenced deterministic tokens and teacher-forced
  distribution, plus sentinel truth, smoothness, memory-growth, and no-OOM
  gates; legacy GGUF/llama consensus is diagnostic only;
- all seven rows pass both phases under their assigned roles; a diagnostic 1k
  win or a single long-row win is not product acceptance.

The machine-readable authority is
`benchmarks/intel-qwen36-35b-a3b-gguf-q4km/acceptance-matrix.json`.
