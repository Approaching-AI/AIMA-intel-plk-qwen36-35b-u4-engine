# Performance target derivation

Snapshot: 2026-07-11

> Historical denominator/feasibility derivation. ADR 0051 and
> `performance-target-2026-07-13.md` supersede its product-matrix scope while
> retaining the measured rows as evidence.

## Decision

The minimum product target is **1.10x the same-host OpenVINO GPU U4
denominator in both prefill and decode**, independently for every accepted
bucket from 1k through 128k. The stretch target is 1.125x. At 256k, where
OpenVINO fails with `CL_OUT_OF_RESOURCES`, the native absolute target is
400 prefill tok/s and 10.00 decode tok/s with no OOM.

This is a product gate, not an average score. A row that wins only prefill or
only decode does not pass, and a route rejection does not complete the goal.

## Measured platform and protocol

The denominator was measured locally on the locked PTL machine: Core Ultra X7
358H, 62 GiB RAM, Arc B390 GPU with 96 compute units at up to 2.5 GHz, OpenCL
driver `26.18.38308.1`, and OpenVINO GenAI
`2026.2.0.0-3121-adf73e80e66`. The OpenVINO model is
`/home/intel/Qwen3.6-35B-A3B-ov`; its IR contains U4 compressed weight
constants and its language-model binary is 18,646,205,498 bytes. The native
model is the locked 21,166,755,168-byte GGUF Q4_K_M file.

The 2026-07-11 calibration lane uses:

- batch size 1, GPU, resident model, prefix caching disabled;
- automatic chat-template insertion disabled;
- exact 1k through 128k input-token buckets;
- `prefill_shape`, `sentinel`, and deterministic `filler` prompt classes;
- one warmup plus three measured generations for every prompt;
- exactly 512 output tokens, with EOS ignored for this fixed-length lane;
- median TTFT/prefill and TPOT/decode per prompt.

Every corrected row checks both the tokenizer count and OpenVINO's
runtime-reported input-token count. This matters because the first 2026-07-11
matrix left `GenerationConfig.apply_chat_template=true`: it counted the raw
prompt before generation but timed a templated prompt. Those bundles remain
useful diagnostics but are not raw-prompt calibration evidence. Seq724-726
disable the template and record the actual pipeline input count.

## Corrected raw-prompt OpenVINO measurements

Each numeric prompt cell below is `prefill tok/s / decode tok/s`, the median of
three measured generations after warmup. Filler is corrected across 1k-128k;
prefill-shape and sentinel are corrected at the 1k/8k anchors. The locked
denominator never decreases: unrefreshed cells retain the harder prior audited
same-host maximum until their raw-prompt replay is complete.

| input | prefill-shape | sentinel | filler | locked denominator |
|---:|---:|---:|---:|---:|
| 1k | 1532.646 / 45.140 | 1538.812 / 45.016 | 1888.091 / 45.253 | 1888.091 / 45.264 |
| 2k | refresh pending | refresh pending | 2206.401 / 44.119 | 2206.401 / 44.226 |
| 4k | refresh pending | refresh pending | 2347.302 / 44.023 | 2347.302 / 44.023 |
| 8k | 2227.211 / 42.359 | 2224.236 / 42.130 | 2281.314 / 42.335 | 2281.314 / 42.359 |
| 16k | refresh pending | refresh pending | 1987.589 / 38.825 | 1987.589 / 38.867 |
| 32k | refresh pending | refresh pending | 1573.042 / 33.773 | 1607.000 / 33.773 |
| 64k | refresh pending | refresh pending | 1116.548 / 26.531 | 1197.000 / 26.531 |
| 100k | refresh pending | refresh pending | 866.537 / 21.648 | 899.000 / 21.648 |
| 128k | refresh pending | refresh pending | 741.607 / 18.806 | 807.990 / 18.806 |

The contract locks the maximum independently for each phase. It also retains
an earlier audited same-host best when that value is higher than the fresh
matrix, so a slower rerun cannot weaken the goal. This affects long-context
prefill and the 32k, 64k, and 128k decode rows shown in the last column.

Corrected evidence bundles:

- prefill-shape/sentinel 1k and 8k: `output/r0-openvino-denominator-matrix-20260711Tseq724-raw-both-cleanZ/`;
- filler 1k and 8k: `output/r0-openvino-denominator-matrix-20260711Tseq725-raw-filler-cleanZ/`;
- filler 2k-128k: `output/r0-openvino-denominator-matrix-20260711Tseq726-raw-filler-rest-cleanZ/`.

The raw-prompt prefill-shape/sentinel 2k-128k rows and the complete 1024-output
lane remain required before a product claim. They may raise the dynamic
same-run threshold or the absolute table, but cannot lower this table.

## Locked target and hardware feasibility

The absolute target is the locked denominator multiplied by 1.10, rounded up
to the next whole prefill token/s and the next 0.01 decode token/s. The strict
decode inventory is 1.975676544 GB of active weights and output head per token,
plus `context * 20,480` bytes of FP16/BF16 KV reads.

| input | denominator prefill / decode | target prefill / decode | strict GB/token | target GB/s |
|---:|---:|---:|---:|---:|
| 1k | 1888.091 / 45.264 | 2077 / 49.80 | 1.997 | 99.433 |
| 2k | 2206.401 / 44.226 | 2428 / 48.65 | 2.018 | 98.157 |
| 4k | 2347.302 / 44.023 | 2583 / 48.43 | 2.060 | 99.745 |
| 8k | 2281.314 / 42.359 | 2510 / 46.60 | 2.143 | 99.885 |
| 16k | 1987.589 / 38.867 | 2187 / 42.76 | 2.311 | 98.828 |
| 32k | 1607.000 / 33.773 | 1768 / 37.16 | 2.647 | 98.354 |
| 64k | 1197.000 / 26.531 | 1317 / 29.19 | 3.318 | 96.848 |
| 100k | 899.000 / 21.648 | 989 / 23.82 | 4.073 | 97.015 |
| 128k | 807.990 / 18.806 | 889 / 20.69 | 4.660 | 96.416 |
| 256k | unavailable | 400 / 10.00 | 7.344 | 73.444 |

The target is meaningful and reachable at the architecture level:

- The measured run-to-run noise floor is 0.5%; a 10% advantage is 20 times
  that band. At 8k/512 it reduces the locked denominator's approximately
  `15.68 s` total to at most `14.25 s`, about `1.43 s` saved while requiring
  both phases to win.
- The machine's raw LPDDR estimate is 136.5 GB/s and the measured planning line
  is 115 GB/s. The strict no-fusion decode inventory now requires only
  `96.416-99.885 GB/s` across 1k-128k. Real packed-Q4 probes already reach
  `108.793-110.522 GB/s`.
- After the accepted Q4-head/exact-Q6-top16 and I8-router/exact-F32-top16 cuts,
  the mixed decode model needs at most `57.690 GB/s` from the remaining real
  Q6_K lane. The measured raw-Q6 full-tensor carrier is `52.720 GB/s`; the
  hard target therefore asks for a bounded `9.43%` carrier improvement, with
  `58 GB/s` as the component kill-number. At the headline 8k row the scaled
  requirement is about `56.79 GB/s`.
- Prefill remains compute-bound. Seq652's correctness-safe, source-realizable
  exact grouped/F16-contribution projection is `9539.674 us` for the real
  layer-27 boundary. The corrected 8k cap is `9526.177 us`: the projection is
  only `13.497 us` (`0.142%`) above it, inside the measured noise band, while
  clean seq673's implemented Q4 boundary reached `9389.725 us` and retained
  `136.452 us` (`1.45%`) headroom. This is architecture feasibility evidence,
  not a promoted full-engine speed row.
- The original 1.25x hard ratio was a planning-roof target. Seq655 replaced its
  proxy assumption with a source-exact native NPU result: the fixed pair is
  capped at `39.376 GB/s` even with zero GPU time, versus the required
  `96 GB/s`. ADR 0014 closes that architecture; ADR 0015 is the explicit
  owner-directed target revision. The 1.125x stretch sits near the passing
  prefill projection boundary and remains aspirational rather than mandatory.

At 256k, 10 decode tok/s implies 73.444 GB/s under the strict byte inventory,
inside the measured engineering band. The 400 prefill tok/s target is a
conservative absolute target until the first successful native 256k row; it
may move upward after evidence, but not downward without an owner-approved
contract revision.

## Product comparison protocol

- batch size 1, GPU, cold prompt/prefix cache, model resident;
- prefix caching disabled for both runtimes;
- one warmup followed by at least three measured generations per prompt;
- three exact prompts per bucket, in both 512- and 1024-output lanes;
- OpenVINO and native rows captured on the same host and software/power state;
- median per prompt and phase, raw TTFT/TPOT distributions, and confirm run;
- native clears `max(absolute target, 1.10 * same-run OpenVINO)` in both
  phases for every bucket, with no averaging across buckets or phases.

The corrected raw-prompt 512-token filler matrix and 1k/8k three-prompt anchors
are complete. The remaining raw prefill-shape/sentinel rows and the 1024-token
denominator lane are mandatory at product promotion and cannot lower the
absolute table. Correctness, reference-consensus deterministic token
equivalence, sentinel retrieval, smoothness, and memory constraints remain
independent hard gates.
