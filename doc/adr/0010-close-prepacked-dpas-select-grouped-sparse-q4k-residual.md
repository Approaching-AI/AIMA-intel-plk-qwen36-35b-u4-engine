# ADR 0010: Close handwritten prepacked DPAS and select grouped sparse Q4_K residual

Date: 2026-07-11

## Status

Accepted; supersedes ADR 0009 as the active route decision.

## Context

ADR 0009 authorized one resident-prepacked native DPAS killer gate after the
exact separated oneDNN route missed the complete routed-MoE cap. Clean seq647
implements that gate on the locked layer-27 capture. It preserves all
`698,351,616` active Q4 codes in `523,763,712` resident bytes and passes the
same three all-value boundaries as seq646:

- SwiGLU: `4,194,304` values, max abs `0.004991293`, RMSE `1.08241e-5`;
- weighted down: `16,777,216` values, max abs `0.000458524`, RMSE
  `2.07396e-6`;
- routed output: `2,097,152` values, max abs `0.000458486`, RMSE
  `5.87822e-6`.

The complete boundary nevertheless takes `19883.191 us` minimum and
`20204.538 us` median versus `8514.926 us`. The stage profile localizes the
failure to the handwritten matrix kernels: gate/up is `12156.248 us` and down
is `6674.270 us`; input quantization, SwiGLU quantization, and scatter together
are only `1110.729 us`. The result is `2.335x` the cap, so bucket, workgroup,
API, GRF, queue, or launch tuning cannot rescue this mapping.

Seq648 profiles the same-host OpenVINO U4 denominator with oneDNN GPU event
timing. Its 40 routed layers do not issue per-expert or power-of-two bucket
GEMMs. Each layer submits exactly three `grouped_gemm:micro` primitives over a
concatenated routed-value buffer and 256 cumulative expert offsets. At the
complete 8k shape (`65,616` routed rows), two gate/up primitives average
`4.870 ms` each and down averages `6.777 ms`.

Seq649 then applies that execution shape to seq639's exact layer-27 offsets:
`222` active experts, `8192` routed rows, and maximum group `361`. Two
F16-by-U4 gate/up primitives plus one down primitive all select
`grouped_gemm:micro` and take `6060.833 us` minimum / `6074.058 us` median.
This is an optimistic performance lower bound, not correctness evidence, but
it leaves `2454.093 us` under the complete cap for Q4_K residual correction,
SwiGLU, weighting, and scatter. The architecture therefore clears the only
valid arithmetic admission test found after seq647.

## Decision

Close `context_wide_1024_prepacked_fused_q4k_dpas_prefill_v1`. Its exact
resident planes remain useful layout evidence, but the handwritten DPAS
mapping and all local variants are closed.

Select `context_wide_1024_grouped_sparse_exact_q4k_residual_prefill_v1` as the
sole active prefill route. Authorize exactly one full layer-27 killer gate:

1. compact all `8192` routed rows by expert and carry a 256-entry cumulative
   offset vector plus deterministic inverse mapping; do not power-of-two pad;
2. retain resident U4 codes and F32 Q4_K scale/min coefficients. The grouped
   core may use an integer U4 zero point, but a fused floating residual must
   restore `scale * q - min` rather than approximate it;
3. execute two grouped gate/up microkernels and one grouped down microkernel,
   with residual correction, SwiGLU, normalized router weighting, and inverse
   scatter inside the same timed boundary;
4. compare all `4,194,304`, `16,777,216`, and `2,097,152` seq646/647 values;
5. keep the complete boundary `<=8514.926 us`. Seq649's `6060.833 us` core
   lower bound leaves only `2454.093 us`; exceeding either correctness or time
   closes the route without a strategy sweep.

oneDNN and OpenVINO are generator and correctness references only. A passing
development gate must be followed by a repository-owned generated kernel or
equivalent native implementation; the promoted runtime cannot link oneDNN or
OpenVINO. `GRPGEMM_USTRATEGY`, datatype, group-count, workgroup, API, GRF,
queue, zero-point approximation, and bucket sweeps are not authorized.

## Consequences

- The route changes dispatch topology, not a launch parameter: one grouped
  sparse primitive replaces hundreds of padded expert-bucket tasks.
- Seq649 proves timing headroom only. Synthetic payloads, integer zero points,
  or the oneDNN dependency cannot satisfy the correctness or product gate.
- F16 routed values are admissible only if all-value and teacher-forced gates
  pass; otherwise the route closes rather than silently weakening accuracy.
- Decode remains unresolved at the `52.79 tok/s` headline floor. No native
  acceptance-matrix row or product speedup is promoted.
- The project goal remains active; this route decision is not completion.

## Evidence

- `output/expert-bucket-dpas-component-gate-20260711Tseq647cleanZ/`
- `output/openvino-onednn-primitive-profile-20260711Tseq648cleanZ/`
- `output/onednn-grouped-u4-moe-preflight-gate-20260711Tseq649cleanZ/`
- `output/onednn-q4k-routed-moe-component-gate-20260711Tseq646cleanZ/`
- `output/prefill-router-shape-census-gate-20260711Tseq639cleanZ/`
- `doc/adr/0009-close-onednn-routed-moe-select-prepacked-fused-q4k-dpas.md`

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
