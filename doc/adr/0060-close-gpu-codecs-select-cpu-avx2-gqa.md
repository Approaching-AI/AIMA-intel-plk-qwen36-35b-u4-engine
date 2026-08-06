# ADR 0060: Close GPU KV codecs and select CPU AVX2 GQA

Date: 2026-07-13

## Status

Accepted. The scaled-E4M3 source and the current GPU compressed-KV route family
are closed. One fixed CPU AVX2/F16C GQA component is selected as an independent
backend feasibility gate. Product promotion remains open.

## Context

Clean seq780 passes finite/numeric and absolute-rate boundaries:

- cosine is `0.999999472859` and relative L2 is `0.001035000497`;
- repeat/confirm seven-sample medians are `2.747708 / 2.767082 ms`, both below
  the fixed `2.825 ms` cap;
- paired median spread is `0.700160%` versus the terminal `0.5%` limit.

The registered decision made noise terminal, so a favorable E4M3 rerun or
sample aggregation change is inadmissible. Together, seq772, seq778, seq779,
and seq780 cover stable-but-slow FP16 scalar, byte-aligned INT8, sub-byte INT6,
and byte-aligned scaled minifloat sources. Further GPU KV datatype or unpack
variants would continue the same stalled axis.

The target CPU is a 16-core Core Ultra X7 358H with AVX2, F16C, FMA, and
AVX-VNNI on the same unified LPDDR. No CPU long-context GQA component has been
measured. Raw FP16 K+V is `256 MiB` per full-attention layer at 128k; the
`2.825 ms` cap corresponds to a `95.02 GB/s` byte-only floor, below the
machine's `136.5 GB/s` raw LPDDR estimate but tight enough to reject the route
immediately if cache sharing or vector compute is inadequate.

## Decision

Close scaled E4M3 and all FP8 format, lookup/storage, scale, group, rounding,
chunk, local-size, subgroup, workgroup, sampling, and integration variants.
Close the current GPU compressed-KV codec axis.

Select exactly one CPU component:

- FP16 K/V, F32 query/gate/output, head dimension `256`, query heads `16`, KV
  heads `2`, GQA group `8`, and context `131072`;
- sixteen pinned worker threads, one query head per worker, with AVX2/F16C/FMA
  score and value loops; GQA peers traverse identical K/V rows concurrently so
  shared-cache/LPDDR reuse is measured rather than assumed;
- online softmax and sigmoid gate with the same F32 reference boundary;
- include current-token F32-to-FP16 K/V conversion and all worker/reduction
  synchronization in wall time; perform zero timed allocation or host transfer;
- five warmups, then two seven-sample distributions and registered medians.

The terminal gate is finite output, cosine `>=0.999`, relative L2 `<=0.002`,
both wall medians `<=2.825 ms/layer`, and paired spread `<=0.5%`. Any numeric,
rate, or noise failure closes CPU thread count, affinity, datatype, vector
width, chunk, softmax form, and synchronization variants without a sweep. A
pass admits 32k/64k guards and heterogeneous packed-backend integration but is
not a product speedup claim.

## Consequences

- Do not rerun or integrate seq780, or open another GPU KV codec.
- This CPU gate is admitted because it changes backend and execution resources,
  not because the E4M3 absolute-rate rows can be cherry-picked.
- Native prefill remains independently closed under ADRs 0048/0050.

Evidence:

- `output/scaled-e4m3-gqa-kv-decode-20260713Tseq780cleanZ/`
- `output/compressed-gqa-i8-kv-decode-20260713Tseq778cleanZ/`
- `output/packed-gqa-i6-kv-decode-20260713Tseq779cleanZ/`
