# ADR 0067: Audit reference-state import before semantic long context

Date: 2026-07-13

## Status

Accepted and completed by ADR 0068. The evidence-only import route is rejected;
native prefill remains closed under ADRs 0048 and 0050.

## Context

Clean seq793-795 prove that fixed hot8192 preserves the hybrid distribution
ladder and clears all seven zero-state decode-capacity guards through 512
measured output tokens. Zero state cannot establish that the engine attends to
a prompt or retrieves a long-context sentinel.

The locked OpenVINO reference exposes all 80 model states through
`InferRequest.query_state()`: 30 conv tensors shaped `[1,8192,4]`, 30 SSM
tensors shaped `[1,32,128,128]`, and ten K/V pairs shaped
`[1,2,context,256]`. The native backend already accepts the corresponding 30
conv, 30 recurrent, and ten K/V pairs through `PackedTokenStateSnapshot`.
Shape compatibility is promising but does not prove layout or numerical
compatibility. In particular, native conv history has three positions and
native K/V is token-major.

Evidence:

- `output/packed-token-level-zero-backend-20260713Tseq793-int8-hot8192-tile4-hostucb-cleanZ/result.json`
- `output/packed-token-context-gap-20260713Tseq795-int8-hot8192-tile4-hostucb-output512-cleanZ/result.json`
- `output/openvino-sdpa-provider-capture-20260713Tseq774cleanZ/raw/worker-result.json`
- `doc/adr/0052-preflight-exact-context-packed-decode-cost.md`

## Decision

Run one bounded reference-state import audit before any semantic long-context
benchmark:

1. Use one locked short prompt and capture all 80 OpenVINO states plus the next
   teacher-forced distribution.
2. Map the 30 logical linear-state indices to the native non-full-attention
   layers and the ten logical K/V indices to native full-attention layers.
3. Derive the conv, recurrent, and K/V transforms from the OpenVINO graph plus
   native state indexing, then use the native CPU state comparison as a numeric
   diagnostic; do not guess or sweep the layout.
4. Import the mapped state into the unchanged hot8192 backend and require the
   existing full-vocabulary `KLD <= 0.005` and top-1-rate `>=0.99` ruler for a
   teacher-forced next step.
5. Any shape, layout, or distribution failure closes this import route without
   a transform sweep. Only a pass admits imported-state sentinel work, starting
   with priority `32k/64k/128k` buckets.

This route validates decode semantics only. It does not allow OpenVINO in the
final runtime, count imported-state construction as native prefill, or
authorize a product performance claim.

## Consequences

- Zero-state decode needs no further optimization while the semantic gate is
  open.
- A passing import gives a practical correctness oracle for long-context
  decode while native-prefill architecture remains a separate blocker.
- A failing import returns the next action to a materially independent native
  prefill/state-construction source with a complete target-facing bound; it
  does not reopen closed transform or kernel variants.

## Follow-Up

- Implement the short state-capture/layout audit and record every state name,
  shape, byte count, transform, and numeric result.
- On pass, build the smallest imported-state sentinel gate for the priority
  long-context buckets before any paired performance matrix.

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
