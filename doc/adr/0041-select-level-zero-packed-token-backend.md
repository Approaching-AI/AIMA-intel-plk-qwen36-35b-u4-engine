# ADR 0041: Select Level Zero for the packed whole-token backend

Date: 2026-07-12

## Status

Accepted; byte census and kernel allocation superseded by ADR 0042

## Context

ADR 0040 selects a persistent packed whole-token schedule. Clean seq732 turns
that route into an O(1)-in-layer-count source contract: 252 ordered logical
commands cover all 693 locked GGUF tensors, the strict 1k stream census is
`1,996,648,064` bytes/token, and the host API exposes one token input and one
top-k output with no intermediate read.

The Arc B390 OpenCL runtime does not advertise `cl_khr_command_buffer`.
Implementing the contract as 252 independent host-side OpenCL enqueue calls
would violate the route's single-submit shape and spend the already narrow
overhead margin on the wrong mechanism. The target does support Level Zero
native modules and reusable regular command lists.

Clean seq733 compiles and validates a PTL zebin, allocates a disjoint device
range for every seq732 command, records 252 kernels plus 251 dependency
barriers once, and submits the closed list once per measured token. Repeat and
confirm stream the exact `1,996,648,064`-byte census in `8.413 / 8.458 ms`
device time (`237.342 / 236.061 GB/s` proxy rate). Minimum host residual is
`9.422 / 10.261 us`; command-list submit itself is below `1 us`. Both rows use
one record, update token control without rerecording, and remain native-only.

Evidence:

- `output/packed-token-schedule-gate-20260712Tseq732cleanZ/`
- `output/packed-token-level-zero-gate-20260712Tseq733cleanZ/`
- `output/product-decode-route-gate-20260712Tseq731cleanZ/`
- `output/native-consensus-gate-20260712Tseq730cleanZ/`

## Decision

1. Select a reusable Level Zero regular command list as the production backend
   mechanism for `resident_packed_full_token_schedule_v5`.
2. Keep the token ID/position in shared control memory so a closed command list
   can be reused without kernel-argument rerecording.
3. Port the accepted real Q4/Q6 carriers and all resident state stages into the
   same list. Load/repack occurs before token timing; only final top-k crosses
   back to the host.
4. Do not treat seq733's byte-stream kernel as model performance. Product
   promotion still requires all real math, seq730 consensus exactness, and a
   timed full-token row within `18.580 ms` device / `20.080 ms` wall.

## Consequences

- OpenCL command-buffer and per-stage host-enqueue implementations are closed
  on this target.
- The Level Zero command-list mechanism and host boundary are no longer the
  blocker; real stage porting and whole-token correctness are.
- The old seq617 Level Zero component was rejected under a bit-exact component
  ruler. It is not promoted here; real whole-token porting remains governed by
  the current component numeric contract and seq730 consensus-token gate.
- Native prefill, sentinel/smoothness, remaining OpenVINO calibration, and the
  full product matrix remain open.

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
