# ADR 0065: Size hot KV for input plus output

Date: 2026-07-13

## Status

Accepted for one source implementation and clean guards. This supersedes ADR
0063's `4096` hot-window size, but not its hybrid representation or ADR 0064's
tile4 algorithm.

## Context

Clean seq787 passes tile4 correctness and clean seq788 passes all seven
twenty-sample exact-context UCB guards. Clean seq789 then executes the complete
512-token loop. Six rows pass, but 4k fails: median/UCB is
`21.082776/21.093802 ms` versus the `20.648358 ms` cap.

The first twenty 4k tokens have a `20.312359 ms` median; the last twenty rise
to `21.143490 ms`, a `0.831131 ms` increase. A 4096-token input exactly fills
hot4096. During generation, each new token evicts one prompt token into the
per-token compressed path, so the short guard no longer exercises the same
all-hot algorithm that cleared ADR 0064's bound.

The product contract is input plus exactly 512 output tokens. The recent-state
tier must be sized against that complete working interval, not input alone.

## Decision

Increase the fixed F32 hot tier to `8192` tokens:

- `8192` is the smallest power of two that covers `4096 + 512`;
- power-of-two ring indexing avoids introducing a non-power-of-two modulo into
  every hot K/V load;
- block32 INT8 for older tokens, FP16 scales, tile4, chunk256, WG256, and
  SIMD32 remain unchanged;
- this is one capacity correction derived from the failed product boundary,
  not a hot-window sweep.

The additional 4096 F32 K/V tokens cost 16 MiB per full-attention layer, or
160 MiB across ten layers. The 128k route remains well below the former F32
state footprint, so this memory cost is admissible for the locked machine.

## Consequences

- Hot4096 is closed for the exact 4k/512 lane even though its twenty-token
  guard passes.
- The hot8192 implementation must rerun clean correctness, all-seven
  twenty-sample guards, and the complete all-seven 512-token lane.
- A 512-token failure closes this fixed capacity choice; no 4608/6144/12288 or
  other window exploration is authorized.
- Zero-state rows still provide cost and capacity evidence only. They do not
  prove sentinel correctness or an OpenVINO speedup.

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
