# ADR 0064: Tile F32 hot KV to clear the short guards

Date: 2026-07-13

## Status

Accepted for clean correctness and exact-context guards. This is not product
promotion.

## Context

Clean seq785 proves ADR 0063's hybrid route clears the absolute decode UCB cap
at 8k through 128k, but misses 2k and 4k by `0.060244` and `0.174296 ms` per
token. Clean seq786 attributes `1.144482/1.256449 ms` to the complete ten-layer
append, partial-attention, and reduction group at those two contexts. Holding
the rest of the token loop fixed requires this group to reach at most
`1.08 ms`.

The hot partial kernel loads one K/V token into SLM, executes all eight GQA
heads, then synchronizes before the next token. Two workgroup barriers per
token mean 512 barriers per chunk256 workgroup. The arithmetic and global
traffic are already bounded; the synchronization cadence is the only local
mechanism large enough to cover the short-row miss without touching the
accepted compressed path.

## Decision

Use one fixed four-token SLM tile for the F32 hot loop:

- each workgroup loads four consecutive K/V tokens into `8 KiB` of SLM;
- the eight query-head subgroups consume those tokens in the original token
  order, preserving online-softmax accumulation order;
- one barrier protects the four-token load and one protects reuse of the tile,
  reducing hot-loop barriers by `4x`;
- compressed INT8 tokens retain the existing per-token decode path;
- block32 INT8, FP16 scales, hot4096, chunk256, WG256, and SIMD32 do not change.

Four tokens is the fixed design point, not a sweep axis. It is the smallest
chosen tile that cuts barrier count by 75% while keeping the complete K+V tile
at only 8 KiB, leaving practical SLM occupancy on Arc B390.

## Consequences

- Development 4k device events reduce the complete ten-layer attention group
  from `1.256449` to `0.758015 ms`, below the registered `1.08 ms` bound.
- Development twenty-sample UCB becomes `20.175706/20.352719 ms` at 2k/4k,
  below `20.554985/20.648358 ms`.
- The teacher-forced ladder remains byte-for-byte equal at maximum KLD
  `0.00389465741953`; a 128k twenty-token diagnostic remains `23.964 tok/s`.
- These development rows authorize only a clean source-bound rerun. No
  output-512 or OpenVINO speedup claim exists.

## Follow-Up

- Commit the fixed tile4 source.
- Run the clean three-case/distribution bundle and all-seven twenty-sample UCB
  guard. Any failed row blocks output512 and closes this fixed design.

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
