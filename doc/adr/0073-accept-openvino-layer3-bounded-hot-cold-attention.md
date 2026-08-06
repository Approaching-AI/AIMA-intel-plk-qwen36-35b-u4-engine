# ADR 0073: Accept the integrated OpenVINO layer-3 bounded hot/cold attention

Date: 2026-07-14

## Status

Accepted

## Context

ADR 0072 accepted the logical request-owned F32-hot8192 plus block32 signed-I8
older-state topology, but not a real attention consumer. Integration exposed
four constraints that the topology-only graph could not settle:

1. The stock layer-3 key Variable supplies both a past-shape source for the
   shared causal mask and a present-shape source used by later full-attention
   reshape/broadcast consumers. Removing only its append path produces invalid
   downstream shapes.
2. Under the default GPU F16 inference policy, graph-F32 SimpleGPU inputs are
   lowered and in-place writes are detached from an F32 Variable. I32 state
   planes preserve the exact IEEE-F32 boundary bits through compilation.
3. A physical 8192-row ring has a decode-wrap race: one workgroup can overwrite
   the evicted slot while other query-head workgroups still read it. One guard
   row gives the current token a unique free slot with no global barrier.
4. The online-softmax arithmetic passed 2k but failed the 8k first-decode
   distribution gate at KLD `0.008563`. A two-pass max then sum/value reduction
   passes that same row at KLD `0.001927`.

Clean seq822 at commit `8bf6bb3` runs isolated stock and candidate workers at
exact 2k and 8k, with one and two same-request decode steps respectively. The
candidate executes one custom and nine stock full-attention operations, has 84
finite states, and contains no stock layer-3 K/V Variable. Its five
teacher-forced KLD rows are `0.000706393`, `0.000096071`, `0.000611830`,
`0.001926514`, and `0.001488872`; stock and candidate greedy paths are exactly
`[271,248068]` and `[271,248068,198]`. Hot values match the stock F16 boundary
bit-for-bit, and cold length `0 -> 1 -> 2`, signed-I8 values, and F16 scale
bytes match the CPU reference exactly.

Evidence:

- `output/openvino-hot-cold-attention-20260714Tseq822-cleanZ/`
- `output/openvino-full-attention-custom-20260714Tseq819-cleanZ/`
- `output/openvino-hot-cold-state-topology-20260714Tseq818-cleanZ/`

## Decision

Use the seq822 carrier as the one-layer semantic base for OV2:

1. The logical hot window remains 8192 tokens. K and V each use request-owned
   I32 state `[1,2,8193,256]` containing the exact IEEE-F32 bits of the stock
   F16 attention boundary; the final row is a physical race-avoidance guard,
   not an additional logical token.
2. Past and present shape vectors are derived from attention-mask and query
   shapes. No replacement K/V history is retained merely to carry length.
3. Older K/V remain append-only signed block32 I8 with byte-exact logical-F16
   scales. A mandatory physical sentinel row makes empty state bindable and
   stores the logical cold length in three base-128 bytes.
4. Fixed-GQA attention dequantizes cold state in the consumer and uses the
   accepted two-pass softmax. It also produces only bounded current-token
   eviction scratch and updates the unique hot guard slot in place.
5. Expansion must use one parameterized source across all ten layers. Per-layer
   source files, cross-request import, property/window/codec sweeps, and a
   return or copy of full hot history are not admitted.

## Consequences

- ADR 0072's logical F32-hot8192 decision remains valid; its literal F32
  `[1,2,8192,256]` physical representation is refined to the I32-bit
  `[1,2,8193,256]` carrier proved here.
- The guard costs 4096 bytes per layer across K and V and removes the need for a
  second update dispatch or cross-workgroup barrier.
- The sentinel is physical metadata and is never attended. Logical cold rows
  begin at physical row one.
- Seq822 closes one-layer arithmetic/state/codec/ownership semantics only. Its
  wall times are diagnostic and it establishes no component or product speedup.
- The next gate is all-ten semantic composition; priority 32k/64k/128k rate
  work remains blocked until that gate passes.

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
