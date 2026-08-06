# intel-qwen36 day-0 R0 plan

Date: 2026-06-26 · R0 closed: 2026-06-27 · Distilled: 2026-06-28

> **This file is the stable R0 bring-up plan, not a progress log.**
>
> - Current state and next action → `STATUS.md` (same directory)
> - Append-only progress narration that used to live inline here →
>   `../../frozen/intel-qwen36-35b-a3b-gguf-q4km/day0-r0-plan-2026-06-26.archive.md`
> - Session timeline → `meta-log/`

## Objective

Bring up the `intel-qwen36` repository according to `meta-engine-factory`:
target contract, oracle, roofline, feasibility gate, and resident harness
adapter before any product-performance work.

## Locked Scope

- Device: Intel PTL CLS DVT2 target, alias `ptl-cls-dvt2-008`
- Model: `Qwen3.6-35B-A3B-GGUF` Q4_K_M
- Batch size: `1`
- Model path: `/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf`
- Model SHA-256:
  `d42becf903ed7093d438c0d9f44afc136756b6b8f766121e9066fb888a2dc36e`

## R0 Gates

1. Refresh `contracts/intel-qwen36-target-contract.json` from the live host.
2. Confirm model file path, byte size, SHA-256, and GGUF metadata.
3. Locate an existing Qwen3.6 GGUF oracle bundle; capture one only if none
   exists.
4. Measure same-host denominator with the same model contract.
5. Measure model-real roofline:
   - low-bit source stream bandwidth
   - M=1 qmatvec shapes
   - prefill projection/attention shapes
   - KV read/write buckets through 256k
6. Compute kill numbers for prefill and decode.
7. Run a cheap feasibility probe for the dominant route.
8. Start resident harness with `load(model, oracle_bundle)`.

## Exit Criteria

R0 closes only when the target, oracle, roofline, feasibility result, and
resident harness status are all recorded under `doc/active/` or `output/`.

If the feasibility probe cannot reach the required byte or matvec class, the
route must be changed before R1 starts.

## Status

R0 gates are **closed**: live target/model facts, oracle bundle validation
(`oracle/r0-oracle-bundle-20260627T060028Z/`), route feasibility (raw OpenCL
GGUF source-stream/qmatvec route **rejected**), and the resident harness load
path. The 262144 denominator and 256k first-token top-k lanes are accepted as
**unavailable** (`doc/adr/0001`, `doc/adr/0002`).

Work has moved to **R1 native correctness**. The live gate, frontier, and next
action are tracked in `STATUS.md` — read that file, not this one, for "what to
do now".
