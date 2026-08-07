# STATUS - intel-qwen36-35b-a3b-gguf-q4km

> Single source of truth for the OPEN GATE + NEXT ACTION only. Machine state ->
> `frontier.json`; decisions -> ledgers/ADRs; timeline -> `meta-log/`.
>
> Snapshot: 2026-08-07

## NEXT ACTION

The resident HTTP technical candidate passes on the bound target. The personal
public repository and its `Approaching-AI` fork are synchronized, Apache-2.0 is
detected on both, and annotated tag `v0.1.0` resolves to public source commit
`f4707fd1af6a87390fc29c104acd5ce6a145c261`. Publish the checksum-verified
runtime, Python wheelhouse, service wheel, and evidence assets as the canonical
fork's GitHub Release, download them externally, and verify `SHA256SUMS`. Do not
reopen seq2300; rerun inference only if runtime payload or service code changes.

## Current Gate

| field | value |
|---|---|
| open gate | **GitHub Release upload and external download verification; bound-target HTTP technical candidate passes** |
| HTTP contract | Models, Completions, Chat Completions, and Responses create/retrieve/delete; JSON/SSE, function tools, structured outputs, byte/entry/TTL-bounded Responses state, bearer auth, bounded-cardinality metrics, request deadlines, graceful drain, and disconnect cancellation |
| context contract | arbitrary caller lengths; smallest internal bucket; no padding/truncation; `prompt_tokens + requested_output_tokens <= max_context_length`; real HTTP checks at 33, 8207, and 131072 prompt tokens |
| resident/cache gate | isolated profile/bucket worker processes; serial batch-1 execution; PID residency check passes; exact prefix-hit tokens match uncached output and bypass reports zero cached tokens |
| identity gate | exact OpenVINO `90214e5be05`, GenAI, and Tokenizers build strings pass before model hashing; plugin build `106` and Python Runtime build `21902` are separate verified identities; full SHA-256 over 12 locked model files / `19,705,459,812 B` reproduces `eb05132e47fe...d7ec`; exact plugin and CONFIG_FILE hashes pass before readiness |
| HTTP evidence | fast suite `66/66`; long/max real smoke `17/17`; final OpenAI Python SDK `2.53.0` smoke `19/19`, covering Models, Completions/Chat/Responses JSON and streams, state lifecycle, and Chat/Responses function tools |
| release artifacts | 42-file RC7 checksum/notices native runtime bundle with Apache-2.0, selected model policy, verified short/long 50-file source postimages, and bit-identical rebuilds; deterministic 14-wheel RC12 offline Python wheelhouse with hash-required install and service wheel `0.1.0`; hardened systemd deployment (`1.7 OK`); zero-finding strict audit over all 10 index-resolvable distributions plus explicit exact-hash/source coverage for non-index artifacts; security/provenance/API docs and release-file scanner |
| publication blockers | GitHub API authorization plus canonical Release upload/download verification only; both public repositories, the exact fork relationship, Apache-2.0, synchronized `main`, and `v0.1.0` tag are externally verified |
| product model | locked `/home/intel/Qwen3.6-35B-A3B-ov` U4 IR; model fingerprint `eb05132e47fe...d7ec` |
| promoted carrier | short profile fingerprint `23f09faa9842...11c` with seq2291 plugin `b63eede5177f...e12`; long profile fingerprint `24aeff1e89e2...3f` |
| formal matrix | all `21/21` bucket/prompt cases, output512, at least eight interleaved ABBA blocks per case |
| paired inference | minimum prefill/decode/total one-sided 95% LCB `1.479464/1.591514/1.581939x`; every 32k/64k/128k phase clears `1.10x`, every shorter phase clears `0.98x` |
| absolute long floors | every required floor passes; minimum prefill/decode margin `209.468759/0.272092 tok/s` |
| correctness | exact greedy output512 tokens in every required row; minimum top-1 `1.0`; maximum KLD `0.004836565 <= 0.005` |
| smoothness | target-normalized prefill/decode CV `0.130518/0.016233`; minimum adjacent retention `1.003732/0.979074`; all `336` jitter rows pass, max P95/P50 `1.162728` |
| isolation and memory | `712` memory rows; max RSS/swap `8,068,968,448/6,544,089,088 B`; minimum available `12,157,624,320 B`; zero OOM/guard events |
| formal evidence | seq2299 gate SHA256 `54a9e432374c...645`; seq2300 rollup SHA256 `2781c586b636...6b9` |
