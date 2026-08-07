# Current Frontier - Tier-3 pointer

> Workstream: `intel-qwen36-35b-a3b-gguf-q4km`
> Snapshot: 2026-08-07

Read this pointer, then `STATUS.md`. This is not a run log.

## Authority map

- Open gate and next action: `STATUS.md`.
- Machine counters and kill-number: `frontier.json`.
- Route decisions: `routes-ledger.json` plus accepted/rejected ledgers.
- Evidence: `meta-log/2026-08-05.md` for seq2300 and
  `meta-log/2026-08-06.md` for the resident HTTP release layer.
- Thresholds: `benchmarks/intel-qwen36-35b-a3b-gguf-q4km/acceptance-matrix.json`.

## Current state

- Seq2300 closes the full product gate with `product_promotion_ready=true` and
  `speedup_claims_allowed=true`. Its deterministic rollup admits all `21/21`
  output512 cases under an exact two-profile bucket policy: seq2291
  affine-Q4/group128 full-logit timing at 2k/4k/8k and the accepted compact
  token-only timing carrier at 16k/32k/64k/128k.
- Every case has at least eight interleaved ABBA blocks. Minimum
  prefill/decode/total one-sided 95% LCB is
  `1.479464/1.591514/1.581939x`; every priority long phase clears `1.10x`,
  every shorter phase clears `0.98x`, and every required long absolute floor
  passes.
- Output512 greedy tokens are exact in every required row, minimum top-1 is
  `1.0`, and maximum KLD is `0.004836565`. Target-normalized prefill/decode CV
  is `0.130518/0.016233`; minimum adjacent retention is
  `1.003732/0.979074`; all `336` jitter rows pass.
- The rollup audits `712` memory rows. Maximum RSS/swap is
  `8,068,968,448/6,544,089,088 B`, minimum available memory is
  `12,157,624,320 B`, and no OOM or guard event occurs.
- ADR 0078 adds a resident OpenAI-compatible release layer without reopening
  the engine gate. Its bound-target technical matrix passes. Upstream model
  identity and stated Apache-2.0 license are evidenced, both promoted plugins
  now have bit-identical standalone source rebuilds, and the irrecoverable
  locked-IR conversion history has an explicit external-artifact boundary.
  The repository owner selected Apache-2.0 and the external exact-hash model
  prerequisite. The personal public repository, exact `Approaching-AI` fork,
  synchronized `main`, and annotated `v0.1.0` tag are externally verified.
  `STATUS.md` owns the remaining canonical GitHub Release upload and external
  checksum verification action.

## Next action

Freeze the promoted carrier and its exact fingerprints. Further optimization
is optional successor work, not an open acceptance gap: admit only a new,
independently fingerprinted, profile-backed kernel route and require the same
complete product gate before it can replace seq2300. For the current HTTP
publication action, follow `STATUS.md` rather than launching another engine
probe.
