# Current Frontier - Tier-3 pointer

> Workstream: `intel-qwen36-35b-a3b-gguf-q4km`
> Snapshot: 2026-08-12

Read this pointer, then `STATUS.md`. This is not a run log.

## Authority map

- Open gate and next action: `STATUS.md`.
- Machine counters and kill-number: `frontier.json`.
- Route decisions: `routes-ledger.json` plus accepted/rejected ledgers.
- Evidence: `meta-log/2026-08-05.md` for seq2300,
  `meta-log/2026-08-06.md` for the resident HTTP release layer, and
  `meta-log/2026-08-09.md` for public Release closure, plus
  `meta-log/2026-08-12.md` for the near-boundary incident and successor gate.
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
  synchronized source, annotated `v0.1.0` tag, canonical GitHub Release, and
  unauthenticated external asset downloads are verified. The downloaded
  payloads and manifest pass the published `SHA256SUMS`; no `v0.1.0`
  publication gate remains.
- A post-release arbitrary-length service run exposed a long-profile LM-head
  physical/logical layout mismatch after a 32-token prefill tail transitions
  to the one-token query. The released `v0.1.0` plugin reproduces the failure
  at 16,380 tokens; the same operator report covers 32,758, 65,519, and
  131,037 tokens. This does not alter the exact-bucket seq2300 evidence, but it
  is a known defect in the broader arbitrary-length service contract.
- The independently fingerprinted `v0.1.1` candidate reinterprets preallocated
  buffers to the active logical layouts. Fixed long-plugin SHA-256 is
  `c0515a401f57...121`. Targeted validation passes all four reproduced lengths,
  maximum context, 67/67 fast tests, 18/18 HTTP smoke, an 8-row
  full-vocabulary comparison with top-1 `1.0` and maximum KLD `0.000092598`,
  and a bit-identical source rebuild. These are maintenance checks, not a new
  formal performance promotion.
- The candidate source and its performance/correctness disclosure are public
  on both Apache-2.0 repositories in commit `af3753db721b...45a`; their current
  `main` heads are synchronized. The existing `v0.1.0` Release now carries a
  prominent warning. No `v0.1.1` tag or Release exists while the successor
  gate is open.

## Next action

Keep `v0.1.0` and seq2300 frozen as historical exact-fingerprint evidence. Run
the complete 21-case output512 ABBA8 performance/correctness/smoothness/memory
gate for fixed long-plugin SHA-256 `c0515a401f57...121`. Only after that passes
may the successor repeat runtime packaging, source reconstruction, security,
annotated tag/Release upload, and anonymous external-download verification.
Do not publish `v0.1.1` or transfer seq2300 speedup claims before those gates.
