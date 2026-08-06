# ADR 0009: Close separated oneDNN routed MoE and select prepacked fused Q4_K DPAS

Date: 2026-07-11

## Status

Accepted; supersedes ADR 0008 as the active route decision.

## Context

ADR 0008 authorized one exact layer-27 oneDNN Q4_K component after seq643's
raw U4 schedule established substantial apparent headroom. Clean seq644
losslessly repacks all `465,567,744` active gate/up codes, applies exact Q4_K
group scales and affine-min compensation, and compares all `4,194,304`
SwiGLU values. It passes at max absolute error `5.007e-6`, RMSE `8.320e-8`,
and cosine `1`; minimum runtime is `4975.470 us` against the `5479.754 us`
component cap. Clean seq645 confirms at `4985.547 us`.

That admission is not a whole-layer result. Seq646 extends the same real
layer-27 schedule through dynamic input gather/Q8, exact gate/up, SwiGLU/Q8,
exact down, normalized router weighting, contribution streaming, and a
deterministic inverse scatter. Only one-time resident weight preparation is
excluded. It losslessly repacks all `698,351,616` active gate/up and down U4
codes and passes three all-value comparisons:

- gate/up through SwiGLU: `4,194,304` values, max abs `0.004991293`,
  RMSE `1.08241e-5`, zero values above `5e-3`;
- weighted down contributions: `16,777,216` values, max abs
  `0.000458524`, RMSE `2.07396e-6`, zero values above `5e-3`;
- final routed MoE output: `2,097,152` values, max abs `0.000458486`,
  RMSE `5.87822e-6`, zero values above `5e-3`.

Correctness is therefore not the blocker. The exact routed boundary takes
`11601.673 us` minimum and `11649.491 us` median against a hard
`8514.926 us` cap. It misses by `3086.747 us` (`1.3625x` the cap), far beyond
the `0.5%` frontier noise band.

The separated-buffer dataflow is the blocker. Its custom kernels alone move
`506,485,248` bytes in the measured implementation; even deleting the entire
input gather leaves `410,522,112` bytes of post-gather compensation,
activation/quantization, contribution, and scatter traffic (`3569.757 us` at
the measured `115 GB/s` planning line). Those streams sit beside 28 oneDNN
JIT-GEMMs and their materialized F32 main/min destinations. Bucket, queue,
post-op, and workgroup tuning cannot remove the defining materialization
boundary.

## Decision

Close `context_wide_1024_onednn_u4_exact_q4k_prefill_v1` as a product route.
Retain seq644/645 as exact component/reference evidence, but do not promote it
to a layer or native prefill claim.

Select `context_wide_1024_prepacked_fused_q4k_dpas_prefill_v1` as the sole
active prefill route. Authorize exactly one real layer-27 fused killer gate:

1. preserve seq639's `222` active experts, all `8192` assignments, and a
   deterministic inverse mapping;
2. prepare resident U4 code, scale, and min planes once, proving every active
   source code is preserved; preparation remains outside the hot timer;
3. consume those planes directly in native DPAS workgroups and accumulate the
   exact group-scale and affine-min terms without materializing separate F32
   main/min matrices;
4. fuse dynamic token Q8, gate/up, SwiGLU-to-Q8, down, normalized router
   weighting, and deterministic scatter in the timed boundary;
5. compare the same `4,194,304`, `16,777,216`, and `2,097,152` values and keep
   the complete boundary `<=8514.926 us`.

The fused gate stops on correctness or timing failure. There is no local-size,
bucket-width, API, GRF, approximation, zero-point, queue, or post-op sweep.
oneDNN remains a codegen/performance reference only and cannot be the final
native runtime dependency.

## Consequences

- Exact gate/up admission did not imply whole routed-MoE feasibility; seq646
  is the required larger boundary and closes that inference gap.
- The next route changes the dataflow, not a launch parameter: compact resident
  planes feed DPAS directly and main/min F32 intermediates disappear.
- The fixed-shape prepack may spend additional resident bytes, but all hot-loop
  code/scale/min reads and dynamic streams remain charged by the `8514.926 us`
  killer cap.
- Approximate integer zero points, S8/BF16 expansion, public oneDNN post-op
  chains, concurrent queues, and finer bucket sweeps are not authorized
  successors to this failure.
- Decode remains unresolved at the `52.79 tok/s` headline floor. No native
  acceptance-matrix row or product speedup is promoted.
- The project goal remains active; this route decision is not completion.

## Evidence

- `output/onednn-q4k-bucket-component-gate-20260711Tseq644cleanZ/`
- `output/onednn-q4k-bucket-component-gate-20260711Tseq645confirmZ/`
- `output/onednn-q4k-routed-moe-component-gate-20260711Tseq646cleanZ/`
- `output/prefill-router-shape-census-gate-20260711Tseq639cleanZ/`
- `doc/adr/0008-close-handwritten-dpas-select-onednn-u4-q4k-component.md`

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
