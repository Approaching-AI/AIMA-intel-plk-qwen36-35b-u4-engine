# ADR 0075: Make the long-context win the product target

Date: 2026-07-14

## Status

Accepted by owner direction. This supersedes the uniform per-bucket ratio in
ADRs 0015, 0051, and 0070; it does not change the locked model, runtime,
correctness reference, or seven-row reporting matrix.

## Context

The uniform `1.10x` requirement at every phase of every row made a short-row
custom-attention win an architectural prerequisite even though the requested
product value is sustained long-context inference. Current evidence shows why
that coupling is counterproductive:

- stock OpenVINO is already an efficient short-context implementation and may
  remain the fastest candidate path at `2k/4k/8k/16k`;
- the accepted bounded-state design addresses traffic that grows with context,
  so its useful separation appears at `32k/64k/128k`, not necessarily before;
- the 128k decode target needs about `96.416-99.885 GB/s`, below the measured
  `108.793-110.522 GB/s` packed-low-bit carrier, so a `1.10x` long-row win is
  aggressive but backed by measured hardware;
- the single-owner tiled prototype fixed a real device-scope publication race
  and can reproduce stock attention closely, but its current unified kernel is
  still materially slower at short contexts. Requiring that kernel to win all
  short rows would select for a local benchmark rather than the intended
  product workload.

The seven exact input lengths and output512 remain important: short rows expose
regressions, while 16k crosses the hot-to-cold state boundary. They need a
non-inferiority contract, not the same optimization quota as the long rows.

Evidence:

- `output/openvino-attention-phase-profile-20260714Tseq839-cleanZ/`
- `output/openvino-hot-cold-attention-20260714Tseq836-allten-cleanZ/`
- `output/compressed-gqa-i8-kv-decode-20260713Tseq783-ci-confirm-cleanZ/`
- `doc/reference/intel-qwen36-35b-a3b-gguf-q4km/performance-target-2026-07-13.md`

## Decision

Keep one exact batch-1, cold-no-prefix, output512 reporting ladder at
`2k/4k/8k/16k/32k/64k/128k`, but assign two explicit performance roles:

1. **Priority win rows — `32k/64k/128k`.** In every priority row, both prefill
   and decode must independently achieve a paired one-sided 95% lower
   confidence bound of at least `1.10x` the same-run isolated stock OpenVINO
   throughput. The existing absolute floors remain additional minima. Because
   both phase ratios pass, the complete output512 latency must also improve by
   at least `1.10x`; report it directly rather than infer a headline from a
   component.
2. **Regression guards — `2k/4k/8k/16k`.** In every guard row, both phases must
   independently prove non-inferiority: the paired one-sided 95% lower bound of
   candidate/stock throughput must be at least `0.98x`. The two-percent margin
   is an explicit maximum product regression budget, not a noise or stability
   estimate. A fixed stock-derived candidate path is allowed and should be
   preferred when a custom path cannot meet this guard.
3. Correctness, deterministic tokens, 16k state-transition truth, sentinel
   retrieval, context smoothness, bounded memory, and no-OOM remain mandatory
   at all seven rows. No averaging may hide a failing priority row, phase, or
   regression guard.
4. The stretch target is `1.125x` in both phases at all three priority rows and
   `1.10x` in both phases at all four guard rows.

This decision does not accept the current carrier, claim a speedup, relax
stock/candidate isolation, or permit a long-row gain to compensate for a guard
that falls below its non-inferiority confidence bound.

## Consequences

- Route selection, profiling, and expensive output512 promotion work start at
  `32k/64k/128k`; 16k remains the mandatory transition gate before them.
- A custom operation no longer has to beat stock SDPA at a short shape. The
  complete bucket-selected candidate still has to pass that row's guard.
- Long-context attention/state work is admitted only when it has a complete
  bound against the priority rows. Short-only launch, wrapper, and submit cuts
  are bundled unless needed to close a measured guard.
- Prefill work prioritizes context-scaling traffic: attention state/layout,
  Transpose plus GatedDeltaNet fusion, DynamicQuantize plus compressed-FC
  reuse, and MoE materialization. Decode work prioritizes bounded hot/cold KV
  state and fused dequantizing attention.
- The seven-row table remains fully reported, so the target change cannot hide
  a short-context cost.

## Follow-Up

- Update the goal, acceptance matrix, status board, and specialization roadmap
  to distinguish priority wins from regression guards.
- Close the single-owner carrier's 2k/8k semantic gate, then prove the exact
  16k state transition before running the priority rows.
- Supersede this ADR only if the owner changes the long-context workload,
  acceptable guard budget, or required stock OpenVINO advantage.

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
