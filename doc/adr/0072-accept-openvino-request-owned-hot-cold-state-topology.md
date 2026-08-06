# ADR 0072: Accept the OpenVINO request-owned hot/cold state topology

Date: 2026-07-14

## Status

Accepted; physical carrier refined by ADR 0073

## Context

ADR 0071 selected same-request F32-hot8192 plus block32 signed-INT8 older K/V,
but its OpenVINO ABI and bounded state movement still had to be proved. Clean
seq811 shows that the dynamic SimpleGPU path supports distinct multiple
outputs; the one-output restriction applies to the legacy static path. Clean
seq812 locks the real layer-3 full-attention Q/K/V/state/mask/scale ABI and
proves exact 2048-to-2049 K/V append semantics in one `InferRequest`.

Generic state update graphs are not bounded on the actual runtime. A dynamic
or trimmed `ScatterElementsUpdate` physically initializes or copies each full
16 MiB hot K/V tensor during decode, despite misleading optimized-out profile
labels. The expected KVCache trim/update fusion is also absent in the actual
compiled pipeline.

Clean seq818 proves an alternative at a 16k-equivalent split. A custom
operation mutates one slot of each request-owned static F32 hot ring and echoes
only the update payload. Append-only I8 KVCache nodes own older K/V and the
exact bytes of their logical F16 block32 scales. Reset, prefill, and decode are
bit-exact; the physical trace contains no full-history update dispatch. The
traced one-layer update kernels total `35.205 us`.

Evidence:

- `output/openvino-multi-output-custom-20260714Tseq811-cleanZ/`
- `output/openvino-full-attention-abi-20260714Tseq812-cleanZ/`
- `output/openvino-hot-cold-state-topology-20260714Tseq818-cleanZ/`

## Decision

Use the following OpenVINO state topology for the OV2 full-attention carrier:

1. K and V each have one request-owned F32 ring of shape
   `[1,2,8192,256]`.
2. A parameterized custom operation mutates only the addressed ring slot and
   returns only the minimal update payload; it does not return or copy the full
   hot history.
3. Immediately after reset, each request-owned hot Tensor is self-bound once
   to mark the static Variable initialized. This transfers no host data and
   does not cross requests or runtimes.
4. Older K/V are append-only signed I8 Variables. Each logical F16 symmetric
   scale per 32 values is stored byte-exact as two I8 bytes, giving scale state
   shape `[1,2,T,16]` for head dimension 256.
5. The next one-source attention operation consumes this topology and fuses
   old-state dequantization with fixed GQA arithmetic. Linear-attention state
   remains on the untouched stock path.

Generic `ScatterElementsUpdate` and an assumed-but-unobserved KVCache trim
fusion are not admitted implementations. This decision accepts state ABI,
ownership, and bounded movement only; it does not accept attention numerics,
distribution correctness, all-layer integration, or product performance.

## Consequences

- Functional in-place mutation is permitted only for request-owned Variable
  input proven by reset/prefill/decode state checks and a physical dispatch
  trace.
- Logical scale precision remains F16 even though its exact physical carrier
  is I8; consumers must reconstruct the two bytes without conversion.
- Future attention kernels must preserve one parameterized source and bounded
  decode movement. A full-ring copy closes that candidate.
- Seq818's `35.205 us` is a topology component measurement and cannot be used
  as an end-to-end speedup claim.

## Follow-Up

- Replace layer 3's real attention boundary on this topology and pass exact
  2k prompt plus decode component/state/codec checks.
- Then run the one-layer teacher-forced distribution and exact greedy-token
  gate before all-ten expansion or `32k/64k/128k` timing.
- Supersede this ADR only if a different same-request topology proves equal or
  better semantics and bounded physical movement.

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
