# ADR 0053: Select fused GQA FP16-KV long-context decode

Date: 2026-07-13

## Status

Accepted as the next native decode component route. Product promotion remains
open on prefill, semantic long-context state, output512, and the full matrix.

## Context

Clean seq770 executes the real 677-kernel packed backend with output512 state
capacity at every core context. The one-token kill-number collapses as context
grows:

| input | native tok/s | target tok/s | target ratio | projected 512 decode |
|---:|---:|---:|---:|---:|
| 2k | 37.534 | 48.65 | 0.772 | 13.641 s |
| 4k | 29.127 | 48.43 | 0.601 | 17.578 s |
| 8k | 19.617 | 46.60 | 0.421 | 26.100 s |
| 16k | 12.288 | 42.94 | 0.286 | 41.665 s |
| 32k | 5.593 | 37.16 | 0.151 | 91.538 s |
| 64k | 3.019 | 29.28 | 0.103 | 169.580 s |
| 128k | 1.567 | 20.69 | 0.076 | 326.820 s |

The rows use zero-initialized state and therefore claim no semantic
correctness or speedup. They do execute the exact context-dependent kernels and
show that no priority bucket clears ADR 0052's 0.80 full-run admission line.

Clean seq771 profiles 128k. The serial
`full_attn_apply_score_gate_control_f32` kernel consumes `542.950 ms/token` and
`full_attn_score_control_f32` consumes `66.205 ms/token`; together they are
`609.155 / 645.671 ms`, or `94.35%` of device time. Host submit is only
`0.029 ms`. The state allocation is `5.081 GiB` because KV is F32, while the
locked bandwidth model assumes FP16/BF16 KV.

## Decision

Close serial F32 score plus per-output-dimension softmax/apply as a
long-context route. Select one fixed GQA-aware FP16-KV decode core that:

- stores K/V in FP16 or BF16 and preserves the model's 2 KV heads / 16 query
  heads / 256 head dimension;
- tiles context across workgroups, computes each score/softmax contribution
  once per query head, and reuses each value tile across its eight GQA heads;
- uses bounded partial output plus a reduction, rather than one serial context
  loop per output dimension;
- keeps all intermediate state on GPU and performs zero timed host transfer;
- matches the existing F32 component at cosine `>=0.999`, relative L2
  `<=0.002`, and finite output.

Retaining seq743's `19.980 ms` non-context device envelope leaves the following
complete full-attention-core cap per layer:

| input | cap per full-attention layer |
|---:|---:|
| 32k | 0.683 ms |
| 64k | 1.407 ms |
| 128k | 2.825 ms |

The first implementation gate is fixed at 128k and `<=2.825 ms/layer` on both
repeat and confirm, with paired spread `<=0.5%`. A miss closes this kernel
shape before integration. Passing 128k admits the 32k/64k guard rows and then
the packed-backend integration; it does not admit output512 or product speed.

## Consequences

- Do not run full output512 on the current serial F32 core.
- Do not sweep context tile, datatype, subgroup, or workgroup before the fixed
  design clears the 128k component cap.
- Prefill remains independently open; a decode component pass cannot complete
  the product goal.

Evidence:

- `output/packed-token-context-gap-20260713Tseq770cleanZ/`
- `output/packed-token-context-gap-20260713Tseq771-profile-128k-cleanZ/`
