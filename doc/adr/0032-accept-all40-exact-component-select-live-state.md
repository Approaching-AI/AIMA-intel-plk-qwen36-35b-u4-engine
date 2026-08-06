# ADR 0032: Accept all-40 exact components and select live-state injection

Date: 2026-07-11

## Status

Accepted for the all-layer routed-MoE component boundary. Live-state,
teacher-forced distribution, deterministic tokens, context, and product speed
remain open.

## Context

ADR 0031 established that the seq673 Q4 and exact-per-16 Q6 payloads fit one
native resident runtime. Clean seq686 then performs the missing execution gate
from one live 1024-token CPU-model evaluation. It captures six real boundaries
for every layer (`240` tensors) and runs all 40 resident handles.

Every per-layer comparison passes finite, cosine `>=0.999`, and relative L2
`<=0.002`. Aggregate evidence covers:

- `167,772,160` SwiGLU values at cosine `0.999999978089`, relative L2
  `0.000209339`;
- `671,088,640` weighted-down values at cosine `0.999999927964`, relative L2
  `0.000379567`;
- `83,886,080` routed outputs at cosine `0.999999933965`, relative L2
  `0.000363415`.

Q4 weighted-down relative L2 spans `0.000223091..0.000512957`; exact-Q6 spans
`0.000210244..0.000630777`. Routed-output ranges are respectively
`0.000215308..0.000496725` and `0.000206419..0.000581088`. One context owns all
`24,746,393,600` real payload bytes, runs 40 handles, and maps no oneDNN or
OpenVINO runtime library.

Complete component times span `8812.043..15265.013 us`; the exact carrier is
therefore still a slow correctness denominator, not a speed result.

Evidence:

- `output/all-layer-exact-q6-component-20260711Tseq686cleanZ/`
- `output/all-layer-exact-q6-prepack-load-20260711Tseq685cleanZ/`

## Decision

Accept the seq673-Q4 plus exact-per-16-Q6 representation at every routed-MoE
component boundary. Do not run more isolated Q4/Q6 component variants.

Select one live-state gate: during a CPU-reference 1024-token transformer
evaluation, execute each resident native routed-MoE component from that layer's
current attention-post-norm, router IDs, and normalized weights; overwrite the
reference `ffn_moe_out` boundary with the native result before downstream
shared-expert, residual, attention, and later-layer work proceeds. Compare each
native output to the same-state CPU boundary before replacement, then compare
the final full-vocabulary distribution against an otherwise identical baseline.

The gate requires:

- exactly 40 ordered native injections with every same-state routed output
  finite, cosine `>=0.999`, relative L2 `<=0.002`;
- final top-1 equality and full-vocabulary KL divergence `<=0.005`;
- one native context, 40 real handles, and native grouped-runtime maps free of
  oneDNN/OpenVINO.

This hybrid reference host is a correctness harness, not the final runtime.
llama.cpp supplies the still-unported graph and oracle; no product speed claim
may use its wall time.

## Consequences

- The all-40 component gate is closed and need not be repeated unless the
  carrier math or representation changes.
- A live-state pass advances to the existing multi-case teacher-forced
  distribution ladder and deterministic tokens.
- A live-state miss is a sequential-drift result; localize the first failing
  injected layer rather than reopening standalone codecs.

## Follow-Up

- Add a callback-only live-injection mode to the existing boundary harness.
- Run paired baseline/injected full-vocabulary rows from a clean commit.

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
