# ADR 0011: Close F16 grouped residual and select in-core affine codegen feasibility

Date: 2026-07-11

## Status

Accepted; supersedes ADR 0010 as the active route decision.

## Context

ADR 0010 authorized one complete layer-27 grouped-sparse exact-Q4_K killer
gate after seq649's optimistic three-core lower bound passed. Clean seq650
implements that boundary on the locked seq646 capture. All three matrix
primitives select `grouped_gemm:micro`, all `698,351,616` active Q4 codes are
repacked losslessly, and the timed path includes gather, floating affine-min
restoration, SwiGLU, down, normalized router weighting, inverse scatter,
submission, and drain.

The complete result fails both terminal gates:

- runtime is `11389.091 us` minimum / `11491.009 us` median versus the
  `8514.926 us` cap (`1.338x` the cap);
- the F16 gate/up destination truncates the main term before the floating
  affine-min residual. The SwiGLU boundary has `109,685 / 4,194,304` values
  above `5e-3`, max abs `0.680302`, and RMSE `0.002089`;
- weighted down and final routed output remain within the component tolerance
  (max abs `0.003062` and `0.003292` respectively), but downstream agreement
  cannot waive the mandatory first boundary.

The synchronized stage profile makes the structural gap explicit. The three
grouped cores alone consume `8247.704 us` (`2492.890 + 2541.425 + 3213.389`),
leaving only `267.222 us` under the complete cap. Gather, external residual
plus activation, down residual plus weighting, and scatter consume another
`3444.096 us` in isolated stage measurements. Public oneDNN attributes cannot
express the required floating Q4_K zero point. A local non-evidence generator
diagnostic that admitted F32 zero points reached a quantization-layout divide
failure and was reverted; it did not produce a runnable kernel.

Consequently, neither public oneDNN composition nor another F16 grouped-value
variant has enough correctness or timing headroom. Changing strategy flags,
datatype, group count, workgroup, API, GRF mode, queue behavior, or zero-point
approximation would be the prohibited sweep from ADR 0010.

## Decision

Close `context_wide_1024_grouped_sparse_exact_q4k_residual_prefill_v1` and all
public-oneDNN / materialized-F16 residual variants.

Select
`context_wide_1024_in_core_exact_q4k_affine_codegen_feasibility_v1` as a new,
architecture-level route. This is not permission to implement or tune a full
kernel. Authorize exactly one static source/ISA and traffic feasibility gate
for a repository-owned grouped microkernel with these invariants:

1. apply Q4_K scale and floating affine-min compensation in the matrix
   accumulator before any F16 store or activation;
2. consume the expert/token mapping without a separately materialized grouped
   input when the traffic model requires that cut;
3. fuse gate/up residual plus SwiGLU into the gate/up epilogue and down
   residual plus router weighting into the down epilogue;
4. avoid materialized F32 gate/up main/min and contribution planes; charge any
   remaining intermediate, inverse-scatter, and synchronization traffic;
5. show native PTL DPAS in the generated ISA and a complete byte/operation
   upper-bound `<=8514.926 us` using measured Arc B390 bandwidth and seq650
   kernel timing, before a real-tensor kernel is authorized.

The feasibility gate is binary. If source inspection shows that the in-core
epilogues cannot preserve F32 accumulator order, or the complete modeled bound
exceeds the cap, close the prefill family without implementation. If it passes,
authorize one real layer-27 kernel under the unchanged three-boundary
correctness and `8514.926 us` gates. oneDNN/OpenVINO may supply generator and
oracle evidence, but the promoted runtime cannot link either dependency.

## Consequences

- Seq650 is a route closure, not a near-pass and not a speed claim.
- F16 materialization before the Q4_K residual is permanently closed; this is
  not reopened by downstream routed-output agreement.
- The successor changes the dataflow boundary by moving exact affine math and
  activation into accumulator epilogues. It is not a oneDNN flag or datatype
  sweep.
- Decode remains unresolved at the `52.79 tok/s` headline floor. No native
  acceptance-matrix row or product speedup is promoted.
- The project goal remains active.

## Evidence

- `output/onednn-grouped-q4k-moe-component-gate-20260711Tseq650cleanZ/`
- `output/onednn-grouped-u4-moe-preflight-gate-20260711Tseq649cleanZ/`
- `output/openvino-onednn-primitive-profile-20260711Tseq648cleanZ/`
- `output/onednn-q4k-routed-moe-component-gate-20260711Tseq646cleanZ/`
- `doc/adr/0010-close-prepacked-dpas-select-grouped-sparse-q4k-residual.md`

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
