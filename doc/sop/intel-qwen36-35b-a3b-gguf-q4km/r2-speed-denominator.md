# SOP — R2 speed denominator

Snapshot: 2026-06-29

R2 is the build-order step that gives perf work a judge. Without it, every
post-R1 ns is self-relative ("faster than my last version") and cannot answer
the only question that matters: *how far from the target*. This SOP builds the
denominator and binds it to the acceptance matrix.

## Where we are (sub-1k smoke)

`tools/intel-qwen36-r2-speed-denominator.py` already aligns the short post-R1
cases against the 1k floor and roofline:

- native decode ≈ **4.2 tok/s** = **0.073× the 1k floor (58 tok/s)** → ~14×
  slower than the same-host reference; **0.34% of the roofline ceiling**
  (1245 tok/s); the 70% bar is 872 tok/s.
- Artifact: `output/r2-speed-denominator-20260628T164203Z/`.

This is a lower bound on a tiny prompt, not the matrix. It already shows the gap
to close is ~14× and lives on the bandwidth axis (consistent with the
bandwidth-roofline reject table), not the dot algorithm.

## The two judges

| judge | source | meaning |
|---|---|---|
| floor | same-host llama.cpp / OpenVINO tok/s | native must beat this to justify existing |
| ceiling | R0 `kv-read-pressure` `ceiling_tok_s_at_qmatvec_max` per bucket | physical roofline; bar = 70% of it |

The floor in `acceptance-matrix.json → bootstrap_targets` is still a placeholder
(`prior 1.30x same-host OpenVINO rows`). **Refresh it before any product claim.**

## Build R2 (on the target)

1. **Native end-to-end matrix.** Extend the native candidate runner to long
   prompts and real decode: `--max-new-tokens 512` over the materialized 1k / 2k
   / 4k / 8k prompt token inputs (the oracle ladder already has these token-id
   rows). Capture per-case `prompt_prefill` and `decode_continuation` ns. Keep
   `cold_no_prefix`, conc=1.
2. **Refresh the same-host floor.** Run `tools/intel-qwen36-r0-llama-denominator-run.py`
   for buckets 1k–8k (these completed in R0 smoke; 262144 stays unavailable per
   ADR 0001). Record prefill + decode tok/s.
3. **Align.** Run `tools/intel-qwen36-r2-speed-denominator.py --diagnostic <new>
   --reference-bucket <b>` per bucket. It emits decode tok/s, `decode_vs_floor`,
   and `decode_roofline_util`.
4. **Bind to acceptance.** Replace `bootstrap_targets` with the refreshed
   same-host floor; clear `is_bootstrap_placeholder`. Bind the manifest to target
   + model + acceptance by path and digest (route guardrail in WORKSTREAMS.md).

## R2 exit gate

R2 closes when, for buckets 1k–8k:

- the native engine runs end-to-end and emits **correct tokens** (R1 gate stays
  closed under teacher-forced check), AND
- every bucket has native prefill/decode **tok/s** bound to the acceptance
  matrix, AND
- the same-host floor is **freshly measured** (not bootstrap), AND
- each row reports `decode_vs_floor` and `roofline_util`.

Only then do R3 optimizations have a denominator. Until then,
`speedup_claims_allowed=false`.

## Reproduce the current smoke alignment

```bash
python3 tools/intel-qwen36-r2-speed-denominator.py
```
