# ADR 0040: Accept consensus-exact decode and select a packed token schedule

Date: 2026-07-12

## Status

Accepted

## Context

ADR 0039 made llama.cpp/OpenVINO consensus a prerequisite for exact candidate
scoring. Clean seq727 evaluates all six canonical token-exact prompts from the
same raw token payloads, with no chat template or prefix cache. Only
`short_transform_003` remains identical for eight generated tokens; the other
references diverge mostly at position 2. The canonical suite therefore cannot
by itself satisfy the three-case prerequisite.

The project already had a 24-prompt, six-domain corpus whose fit/validation/test
split was locked before tokenization and before any reference result. Clean
seq728 measures all 24 rows for eight tokens and clean seq729 extends the same
full corpus to nine tokens. Nine cases agree exactly between llama.cpp and
OpenVINO at both lengths, including fit, validation, and test rows.

Clean seq730 binds three pre-registered consensus rows—one per split—to the
accepted rowblock16 26-layer GPU decode carrier. The CPU/llama prefill supplies
the first seed token; each subsequent eight-token GPU sequence exactly equals
positions 1..8 of both nine-token references. Median diagnostic decode is
`19.40189718 tok/s`. This is an exact decode-carrier result, not native prefill
or a product speed row.

Applying the actual 1k product floor changes the route decision. The paired
valid profile records `50.556 ms/token` wall, `36.354 ms/token` current-kernel
floor, and `14.202 ms/token` non-kernel overhead. The `49.80 tok/s` target
allows only `20.080 ms/token`. Even deleting all overhead yields only
`27.507 tok/s`; current kernels need at least a `44.76%` cut.

Real full-tensor carrier evidence makes a structural route credible: packed Q4
reaches `110.522 GB/s` and exact-Q6 rowstripe reaches `107.579 GB/s`, both above
the strict 1k requirement of `99.433 GB/s`.

Evidence:

- `output/reference-consensus-matrix-20260712Tseq727cleanZ/`
- `output/reference-consensus-matrix-20260712Tseq729-fresh9-cleanZ/`
- `output/native-consensus-gate-20260712Tseq730cleanZ/`
- `output/product-decode-route-gate-20260712Tseq731cleanZ/`

## Decision

1. Accept seq729 as the reference-consensus corpus and seq730 as the exact
   post-seed native decode carrier gate.
2. Retire the `19.5 tok/s` Vulkan bring-up floor. The active short-context
   product floor is `49.80 tok/s`.
3. Select `resident_packed_full_token_schedule_v5`: preserve the measured Q4
   and exact-Q6 packed carriers, schedule the whole token persistently, and
   remove inter-stage host drains across at least selected FFN, linear preconv,
   and attention front.
4. Admit implementation only with a full-token kernel schedule
   `<=18.580 ms/token`, residual host/submit overhead `<=1.500 ms/token`, full
   wall `<=20.080 ms/token`, strict bandwidth `>=99.433 GB/s`, and unchanged
   component plus seq730 consensus exactness.

## Consequences

- Exact-token correctness is no longer the immediate blocker for native
  post-seed decode.
- The current rowblock26 lane remains an accepted correctness carrier but is
  rejected as a product-speed architecture.
- Overhead microcuts, isolated stage tweaks, local-size variants, and codec or
  layer-mask sweeps are closed before a whole-token schedule design proof.
- Native prefill, context/sentinel/smoothness, remaining OpenVINO calibration,
  and the full product matrix remain open.

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
