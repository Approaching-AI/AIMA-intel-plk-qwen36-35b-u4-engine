# ADR 0006: Close sparse-exception Q6 and select a grouped-prefill census

Date: 2026-07-11

## Status

Superseded by ADR 0007 after real 64-token routing failed the gate/up
weight-memory lower bound.

## Context

Clean seq637 executes ADR 0005's one exact-Q6 source-format gate over all 60
non-head Q6 tensors. It scans `4,619,059,200` source bytes and reconstructs
every Q6 code with zero mismatches. Expert-down traffic is charged using the
worst encoded eight experts per tensor, giving the locked active source
inventory of `352,665,600` bytes.

The proposed contiguous signed 4-bit window is not sparse on the real model.
It needs `144.952` exceptions per active 256-code block. Directories,
alignment, headers, base codes, scale metadata, and exceptions expand active
traffic to `743,794,928` bytes, or `2.10907x` raw Q6.

Even an impossible zero-compute, planning-line stream would reach only
`54.5265 GB/s` in original-Q6-equivalent throughput. The preregistered
optimistic additive model—base bytes at the measured packed-Q4
`110.522 GB/s`, auxiliary bytes at `115 GB/s`, and exception MACs at the
packed-Q4 value rate—predicts `45.2537 GB/s` and `7.79359 ms` for the 8k Q6
lane. The gate is `96 GB/s` and `3.74439 ms`. This is below the kill-number
and slower than the existing raw-Q6 `52.7204 GB/s` carrier before a kernel is
written.

## Decision

Close `q6_sparse_exception_repack_feasibility_v1` without implementing a GPU
kernel. Also close contiguous-window exception-threshold, directory,
alignment, workgroup, and API variants; none can overcome the memory-only
ceiling.

Pop the highest-ranked independent parked route:
`target_facing_grouped_dpas_prefill_v2`. Its first unit is a real-router shape
census, not another kernel:

1. capture exact top-8 expert assignments for real 64-token tiles from the
   locked prompt materialization, preserving layer and token identity;
2. report per-layer expert group-size histograms, active expert counts, weight
   reuse, and dispatch shapes for selected gate/up and down;
3. derive an optimistic grouped-GEMM roof against the current 8k whole-layer
   budget of `575.33 us`;
4. compare directly with the existing selected gate/up-to-SwiGLU baseline
   `1470.833 us`, which needs at least `2.556x` merely to fit the entire layer
   budget.

Only a census-backed design that clears that necessary bound may authorize
one real routed-token MxN DPAS component. Repeated-token rowlane, private token
loops, local-Q8 sharing, and occupancy-only variants remain closed.

## Consequences

- The exact sparse-exception codec is rejected by arithmetic; no GPU time is
  spent confirming an impossible format.
- This decision does not reopen fixed low4/high2 split-plane Q6, persistent I8
  expansion, or derived Q5.
- Prefill evidence must use real router assignments. A synthetic repeated
  token group is not batch-1 MoE reuse evidence.
- The census is shape/roofline evidence only. It is not a prefill speed claim.
- Decode still has no accepted architecture at the `52.79 tok/s` product
  floor. Advancing the independent prefill gate does not waive that gap.
- The project goal remains active; route rejection is not completion.

## Evidence

- `output/q6-sparse-exception-gate-20260711Tseq637cleanZ/`
- `output/gpu-q4x8-qmatvec-ffn-gateup-full-20260702T225500Z/`
- `output/gpu-q6-qmatvec-layer7-ffn-down-full-20260702T234500Z/`
- `output/dpas-storage-workdist-gate-20260707Tseq79Z/`

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
