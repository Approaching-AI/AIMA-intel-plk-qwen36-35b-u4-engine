# STATUS - intel-qwen36-35b-a3b-gguf-q4km

> Single source of truth for the OPEN GATE + NEXT ACTION only. Machine state ->
> `frontier.json`; decisions -> ledgers/ADRs; timeline -> `meta-log/`.
>
> Snapshot: 2026-08-12

## NEXT ACTION

Keep Apache-2.0 `v0.1.0` and seq2300 frozen as historical exact-fingerprint
results. The released long plugin has a reproduced arbitrary-length service
defect at the 32-token-prefill to one-token-query transition. The `v0.1.1`
release-candidate reinterpretation fix passes targeted 16,380/32,758/65,519/
131,037-token performance, boundary correctness, 67/67 fast tests, 18/18 real
HTTP smoke, maximum context, and bit-identical source rebuild checks. Do not
publish or transfer the seq2300 speedup claim to the successor fingerprint.
Run the complete 21-case output512 ABBA8 performance, correctness, smoothness,
and memory gate for the fixed plugin; then repeat packaging, security, source,
runtime, canonical Release, and anonymous-download verification.

## Current Gate

| field | value |
|---|---|
| open gate | **v0.1.1 successor promotion — the complete product and publication gate must be repeated for fixed long-plugin SHA-256 `c0515a40...121`** |
| frozen release | `v0.1.0` remains published at source tag commit `f4707fd1af6a87390fc29c104acd5ce6a145c261`; its formal seq2300 results apply only to the released fingerprints |
| known v0.1.0 defect | arbitrary long lengths can reach a preallocated physical/logical LM-head layout mismatch; released plugin `01c04ced...269` reproduces the exact failure at 16,380 tokens |
| fixed carrier | long plugin `c0515a401f57...121`; source state `77153ecf9ed7...067`; three-file delta `017f5eb4925c...ed`; standalone rebuild is bit-identical |
| public source | personal and canonical Apache-2.0 public `main` branches are synchronized and contain candidate source/evidence commit `af3753db721b40d28b30773468d4b0c10d1cb45a`; `v0.1.0` Release notes carry the known-issue warning; no `v0.1.1` tag or Release exists |
| HTTP contract | Models, Completions, Chat Completions, and Responses create/retrieve/delete; JSON/SSE, function tools, structured outputs, byte/entry/TTL-bounded Responses state, bearer auth, bounded-cardinality metrics, request deadlines, graceful drain, and disconnect cancellation |
| context contract | arbitrary caller lengths; smallest internal bucket; no padding/truncation; `prompt_tokens + requested_output_tokens <= max_context_length`; real HTTP checks at 33, 8207, and 131072 prompt tokens |
| resident/cache gate | isolated profile/bucket worker processes; serial batch-1 execution; PID residency check passes; exact prefix-hit tokens match uncached output and bypass reports zero cached tokens |
| identity gate | exact OpenVINO `90214e5be05`, GenAI, and Tokenizers build strings pass before model hashing; plugin build `106` and Python Runtime build `21902` are separate verified identities; full SHA-256 over 12 locked model files / `19,705,459,812 B` reproduces `eb05132e47fe...d7ec`; exact plugin and CONFIG_FILE hashes pass before readiness |
| v0.1.1 targeted HTTP evidence | fast suite `67/67`; real smoke `18/18`, including JSON/SSE, all four reproduced near-boundary lengths with status 200, and exact 131,072-token maximum context |
| v0.1.1 targeted performance | controlled single-run output64 rows at 16,380/32,758/65,519/131,037 tokens: TTFT `15.200/76.482/104.199/185.026 s`; inter-token decode `46.49/39.58/30.41/21.09 tok/s`; maintenance evidence only, no successor speedup claim |
| v0.1.1 targeted correctness | 16,380-token full-vocabulary teacher-forced comparison: top-1 `8/8`, maximum KLD `0.000092598 <= 0.005`; all four output64 runs finite; complete output512 successor matrix remains open |
| v0.1.1 publication blockers | full 21-case ABBA8 product gate, successor smoothness/memory rollup, final runtime/wheel/evidence packaging, security audit, annotated tag/canonical Release upload, and anonymous checksum verification |
| product model | locked `/home/intel/Qwen3.6-35B-A3B-ov` U4 IR; model fingerprint `eb05132e47fe...d7ec` |
| frozen v0.1.0 formal matrix | all `21/21` exact-bucket prompt cases, output512, at least eight interleaved ABBA blocks per case |
| frozen v0.1.0 paired inference | minimum prefill/decode/total one-sided 95% LCB `1.479464/1.591514/1.581939x`; applies only to the released seq2300 fingerprints |
| frozen v0.1.0 correctness | exact greedy output512 tokens in every required row; minimum top-1 `1.0`; maximum KLD `0.004836565 <= 0.005` |
| frozen v0.1.0 formal evidence | seq2299 gate SHA256 `54a9e432374c...645`; seq2300 rollup SHA256 `2781c586b636...6b9` |
