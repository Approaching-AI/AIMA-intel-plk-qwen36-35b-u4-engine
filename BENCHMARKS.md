# v0.1.0 Performance and Correctness Report

This report publishes the performance, inference-equivalence correctness, and
stability results used to promote `v0.1.0`. The detailed machine-readable
records are in the checksum-bound
[`v0.1.0` evidence archive](https://github.com/Approaching-AI/AIMA-intel-plk-qwen36-35b-u4-engine/releases/download/v0.1.0/AIMA-intel-plk-qwen36-35b-u4-engine-evidence-0.1.0.tar.zst).

## Result

The release passes all `21/21` required cases: seven prompt-length buckets,
three prompt classes per bucket, and exactly 512 generated tokens per case.
The candidate beats the locked same-host stock OpenVINO GPU reference in every
reported phase after applying the paired one-sided 95% lower confidence bound
(LCB).

| Gate | v0.1.0 result | Required | Status |
|---|---:|---:|:---:|
| Required cases | 21/21 | 21/21 | pass |
| Minimum prefill candidate/stock LCB | 1.479464x | 1.10x priority; 0.98x guard | pass |
| Minimum decode candidate/stock LCB | 1.591514x | 1.10x priority; 0.98x guard | pass |
| Minimum total candidate/stock LCB | 1.581939x | diagnostic summary | pass |
| Exact greedy tokens | 10,752/10,752 | every token | pass |
| Teacher-forced top-1 agreement | 1.000000 minimum | at least 0.99 | pass |
| Teacher-forced KLD | 0.004836565 maximum | at most 0.005 | pass |
| Long-context sentinel retrieval | 7/7 candidate and stock cases | every bucket | pass |

These are results for the locked target and workload below. They are not a
claim about other Intel GPUs, other models, other precisions, batching, or
prefix-cache-hit traffic.

## Tested Configuration

| Field | Value |
|---|---|
| Release | `v0.1.0`; source commit `f4707fd1af6a87390fc29c104acd5ce6a145c261` |
| Date of promoted rollup | 2026-08-05 (`seq2300`) |
| Hardware | Intel PTL CLS DVT2, Core Ultra X7 358H, Intel Arc B390 GPU, 64 GB-class LPDDR |
| OS | Ubuntu 24.04.4 LTS, kernel `6.17.12-061712-generic` |
| Model | locked external Qwen3.6-35B-A3B OpenVINO U4 IR |
| Model fingerprint | `eb05132e47fe...d7ec`; full 12-file identity is in the model contract and evidence |
| OpenVINO Runtime | `2026.2.0-21902-90214e5be05-releases/2026/2` |
| OpenVINO GenAI | `2026.2.0.0-3121-adf73e80e66` |
| Baseline | immutable same-host stock OpenVINO GPU U4 worker |
| Candidate | promoted short/long-profile OpenVINO GPU specialization |
| Workload | batch 1, cold/no-prefix, 2k through 128k prompt buckets, output 512 |
| Prompt classes | filler, prefill-shape, and long-context sentinel |
| Schedule | at least eight interleaved candidate/stock ABBA blocks per case |
| Decision statistic | deterministic 20,000-resample percentile bootstrap of the paired median ratio; one-sided 95% LCB |

The stock and candidate workers use separate plugin/cache/configuration paths.
The baseline does not load candidate custom-operation configuration.

## Performance

### Bucket summary

Candidate throughput is the median across the three prompt classes in the
bucket, as recorded by the smoothness ladder. Each LCB column is the worst
paired candidate/stock result among those three cases, so the table does not
hide a weak prompt class behind an average.

| Prompt bucket | Candidate prefill tok/s | Candidate decode tok/s | Worst prefill LCB | Worst decode LCB | Worst total LCB | Gate | Status |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 2k | 2105.22 | 51.66 | 1.479464x | 1.591514x | 1.581939x | 0.98x | pass |
| 4k | 2367.89 | 50.57 | 1.568074x | 1.617308x | 1.610465x | 0.98x | pass |
| 8k | 2462.30 | 47.96 | 1.654805x | 1.600109x | 1.618982x | 0.98x | pass |
| 16k | 2337.37 | 45.84 | 1.648845x | 1.676622x | 1.675392x | 0.98x | pass |
| 32k | 2065.28 | 38.84 | 1.608355x | 1.679497x | 1.647822x | 1.10x | pass |
| 64k | 1621.72 | 30.25 | 1.600691x | 1.767016x | 1.656342x | 1.10x | pass |
| 128k | 1098.78 | 21.01 | 1.813360x | 1.890407x | 1.834001x | 1.10x | pass |

The 2k-16k rows are formal non-regression guards at `0.98x`; the 32k-128k
priority rows require `1.10x` in both prefill and decode. The observed LCBs
also exceed `1.10x` in every short-bucket case.

### All paired cases

| Case | ABBA blocks | Prefill LCB | Decode LCB | Total LCB | Absolute floors |
|---|---:|---:|---:|---:|:---:|
| `filler_002k` | 8 | 1.527973x | 1.625415x | 1.617825x | pass |
| `prefill_shape_002k` | 8 | 1.480392x | 1.591514x | 1.581939x | pass |
| `sentinel_002k` | 8 | 1.479464x | 1.609746x | 1.598783x | pass |
| `filler_004k` | 8 | 1.596465x | 1.633058x | 1.628377x | pass |
| `prefill_shape_004k` | 8 | 1.568074x | 1.624408x | 1.616597x | pass |
| `sentinel_004k` | 8 | 1.568558x | 1.617308x | 1.610465x | pass |
| `filler_008k` | 8 | 1.654805x | 1.624798x | 1.629884x | pass |
| `prefill_shape_008k` | 8 | 1.672002x | 1.620779x | 1.633280x | pass |
| `sentinel_008k` | 8 | 1.675442x | 1.600109x | 1.618982x | pass |
| `filler_016k` | 8 | 1.692246x | 1.688890x | 1.692995x | pass |
| `prefill_shape_016k` | 8 | 1.648845x | 1.687424x | 1.675392x | pass |
| `sentinel_016k` | 8 | 1.670643x | 1.676622x | 1.679877x | pass |
| `filler_032k` | 8 | 1.637009x | 1.679497x | 1.667594x | pass |
| `prefill_shape_032k` | 8 | 1.623239x | 1.706652x | 1.658345x | pass |
| `sentinel_032k` | 8 | 1.608355x | 1.701856x | 1.647822x | pass |
| `filler_064k` | 8 | 1.675892x | 1.806065x | 1.722238x | pass |
| `prefill_shape_064k` | 8 | 1.654718x | 1.767016x | 1.699132x | pass |
| `sentinel_064k` | 8 | 1.600691x | 1.790880x | 1.656342x | pass |
| `filler_128k` | 8 | 1.891520x | 1.932479x | 1.932822x | pass |
| `prefill_shape_128k` | 8 | 1.813360x | 1.890407x | 1.834001x | pass |
| `sentinel_128k` | 8 | 1.985347x | 1.915885x | 1.987704x | pass |

`Total LCB` is the paired end-to-end prompt-plus-generation ratio. Promotion
is decided per bucket and per phase; phase or bucket averaging is not allowed.

## Correctness Eval

Correctness here means inference equivalence to the locked stock OpenVINO GPU
U4 reference. Every case performs deterministic greedy generation and a
teacher-forced comparison at all 512 output positions. In total, the gate
checks 10,752 generated tokens and 10,752 teacher-forced distributions.

| Case | Max per-position KLD | Top-1 agreement | Greedy tokens |
|---|---:|---:|:---:|
| `filler_002k` | 0.000888745 | 1.000000 | 512/512 exact |
| `prefill_shape_002k` | 0.004836564 | 1.000000 | 512/512 exact |
| `sentinel_002k` | 0.001672847 | 1.000000 | 512/512 exact |
| `filler_004k` | 0.000495818 | 1.000000 | 512/512 exact |
| `prefill_shape_004k` | 0.000916086 | 1.000000 | 512/512 exact |
| `sentinel_004k` | 0.001529916 | 1.000000 | 512/512 exact |
| `filler_008k` | 0.000333602 | 1.000000 | 512/512 exact |
| `prefill_shape_008k` | 0.003760504 | 1.000000 | 512/512 exact |
| `sentinel_008k` | 0.001138390 | 1.000000 | 512/512 exact |
| `filler_016k` | 0.000152933 | 1.000000 | 512/512 exact |
| `prefill_shape_016k` | 0.000735058 | 1.000000 | 512/512 exact |
| `sentinel_016k` | 0.002081355 | 1.000000 | 512/512 exact |
| `filler_032k` | 0.000104148 | 1.000000 | 512/512 exact |
| `prefill_shape_032k` | 0.000479658 | 1.000000 | 512/512 exact |
| `sentinel_032k` | 0.000819746 | 1.000000 | 512/512 exact |
| `filler_064k` | 0.000055777 | 1.000000 | 512/512 exact |
| `prefill_shape_064k` | 0.000253623 | 1.000000 | 512/512 exact |
| `sentinel_064k` | 0.003761955 | 1.000000 | 512/512 exact |
| `filler_128k` | 0.000042709 | 1.000000 | 512/512 exact |
| `prefill_shape_128k` | 0.000064959 | 1.000000 | 512/512 exact |
| `sentinel_128k` | 0.004424813 | 1.000000 | 512/512 exact |

The seven sentinel prompts pass retrieval under both the candidate and stock
workers. Distribution rows are finite and complete, and any first greedy-token
divergence would block promotion.

The evidence also records a raw full-logit-vector cosine diagnostic. It is
explicitly non-gating and does not meet the component-level `0.999` threshold
(minimum `0.516088`). This is disclosed rather than silently omitted. The
product correctness contract instead gates the final model output on exact
greedy tokens, top-1 agreement, normalized-distribution KLD, and sentinel
retrieval; all of those required checks pass. The `0.999` component cosine
threshold remains applicable to separately isolated component tests.

This release does **not** publish MMLU, GSM8K, HumanEval, IFEval, or another
model-capability score. Those suites measure the model's knowledge and task
quality; this eval measures whether this specialized engine preserves the
locked model's behavior relative to stock OpenVINO.

## Context Smoothness, Jitter, and Memory

| Check | Result | Required | Status |
|---|---:|---:|:---:|
| Target-normalized prefill CV | 0.130518 | at most 0.15 | pass |
| Target-normalized decode CV | 0.016233 | at most 0.12 | pass |
| Minimum adjacent prefill retention | 1.003732 | at least 0.75 | pass |
| Minimum adjacent decode retention | 0.979074 | at least 0.75 | pass |
| Candidate jitter rows | 336/336 pass | every row | pass |
| Maximum decode TPOT P95/P50 | 1.162728 | at most 1.25 | pass |
| Memory observations | 712 | full request lifetime | pass |
| Maximum process RSS | 8,068,968,448 B | no OOM/guard event | pass |
| Maximum process swap | 6,544,089,088 B | no OOM/guard event | pass |
| Minimum system memory available | 12,157,624,320 B | no OOM/guard event | pass |
| OOM or memory-guard events | 0 | 0 | pass |

## Service Validation

The released OpenAI-compatible service separately passes:

- `66/66` fast service tests;
- `17/17` real HTTP long/max-context smoke checks, including a 131,072-token
  prompt;
- `19/19` OpenAI Python SDK `2.53.0` checks across Models, Completions, Chat,
  Responses, JSON/SSE streaming, state lifecycle, and function tools.

These are API and operational acceptance results. They are not substituted for
the paired performance or inference-equivalence gates above.

## Audit the Raw Evidence

The release source tree intentionally excludes multi-megabyte raw experiment
outputs. Download the evidence asset and verify its SHA-256 before extracting:

```bash
curl -LO https://github.com/Approaching-AI/AIMA-intel-plk-qwen36-35b-u4-engine/releases/download/v0.1.0/AIMA-intel-plk-qwen36-35b-u4-engine-evidence-0.1.0.tar.zst
printf '%s  %s\n' \
  1b34014ab5ab000b10074936dad506f0a9c76dd59c3e18a6495742785c944349 \
  AIMA-intel-plk-qwen36-35b-u4-engine-evidence-0.1.0.tar.zst \
  | sha256sum -c -
tar --zstd -xf AIMA-intel-plk-qwen36-35b-u4-engine-evidence-0.1.0.tar.zst
```

The promoted rollup is under
`output/openvino-affine-q4-product-rollup-20260805Tseq2300-clean/`:

- `performance.json` contains every paired timing row and confidence bound;
- `correctness.json` contains per-token equality and distribution checks;
- `smoothness.json` contains the bucket ladder and 336 jitter rows;
- `memory.json` contains the 712 memory observations;
- `gate.json`, `manifest.json`, and `summary.md` bind the promotion decision;
- `MANIFEST.sha256` binds every file in the public evidence archive.

The governing thresholds are versioned in
[`benchmarks/intel-qwen36-35b-a3b-gguf-q4km/acceptance-matrix.json`](benchmarks/intel-qwen36-35b-a3b-gguf-q4km/acceptance-matrix.json),
and the locked model/runtime identities are in
[`contracts/qwen36-35b-a3b-openvino-u4-model-contract.json`](contracts/qwen36-35b-a3b-openvino-u4-model-contract.json).
