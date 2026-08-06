# ADR 0061: Record long-context native route exhaustion

Date: 2026-07-13

## Status

Accepted pending an owner contract decision or independently verified new
capability. The project goal remains active and the performance target is not
lowered.

## Context

Clean seq781 executes ADR 0060's independent CPU backend exactly as registered:

- all sixteen workers are pinned; AVX2/F16C/FMA, current-token conversion, and
  persistent-worker synchronization are active;
- cosine is `0.999999999995`, relative L2 is `0.000149519232`, and output is
  finite;
- repeat/confirm seven-sample wall medians are
  `31.563737 / 31.697967 ms` versus the `2.825 ms` cap;
- paired spread passes at `0.423466%`.

The faster CPU row is `11.17x` over budget, so no thread, affinity, vector,
chunk, or synchronization variant can rescue the backend. NPU decode and its
graph/compiler/precision variants remain closed by the earlier exact-Q6 source
decision and ADR 0049; ADR 0050 also explicitly retains that closure.

The long-context search now has terminal evidence for every admissible native
family under the locked contracts:

- serial F32 GPU attention is an order of magnitude too slow (seq770/771);
- scalar FP16 and fixed XMX GPU distributions fail rate (seq772/773);
- stateful and paged provider programs fail rate or stability (seq775/777);
- byte-aligned INT8 and E4M3 clear absolute rate but fail their registered
  noise gates; packed INT6 fails rate (seq778-780);
- CPU AVX2/F16C passes accuracy/stability but fails rate by `11.17x` (seq781);
- NPU decode is already closed independently.

Native prefill is also still closed under ADRs 0048/0050 because the unchanged
8k hard guard has no complete source-derived route below its cap. Therefore no
remaining source can produce the required seven-bucket, dual-phase output-512
matrix without reopening a terminal decision.

## Decision

Close `native_cpu_avx2_fp16_gqa_decode_v15` and record long-context native route
exhaustion under the simultaneous locked hardware, model/precision, batch-1,
correctness, native-runtime, seven-bucket output-512, and `1.10x` OpenVINO
contracts.

Do not launch another attention datatype, codec, GPU kernel distribution,
provider property, CPU thread/vector variant, NPU graph, or current-source
prefill replay. A successor requires one of:

1. an owner-recorded change to hardware, model/KV precision contract,
   correctness tolerance, batch size, final runtime dependency, bucket/phase
   acceptance shape, or minimum OpenVINO speedup ratio; or
2. an independently verified new compiler/hardware capability with a complete
   source-derived bound below the relevant cap before implementation.

This is route exhaustion, not project completion, product acceptance, or a
speedup claim. Seq778/780's favorable absolute medians cannot be promoted by
discarding their terminal noise failures.

## Consequences

- `STATUS.md` moves to an owner-contract decision gate.
- The quantitative long-context target remains authoritative and unchanged.
- All existing code/evidence stays reproducible, but none is a promoted
  output-512 product runtime.

Evidence:

- `output/cpu-avx2-fp16-gqa-decode-20260713Tseq781cleanZ/`
- `output/scaled-e4m3-gqa-kv-decode-20260713Tseq780cleanZ/`
- `output/compressed-gqa-i8-kv-decode-20260713Tseq778cleanZ/`
- `doc/adr/0048-record-measured-1p10-prefill-route-exhaustion.md`
