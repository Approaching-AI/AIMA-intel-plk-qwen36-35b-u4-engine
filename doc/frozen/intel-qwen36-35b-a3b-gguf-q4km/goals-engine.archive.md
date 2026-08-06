> **ARCHIVED — DO NOT EDIT.** Frozen snapshot of append-only progress narration,
> moved out of the live docs during the 2026-06-28 documentation distillation.
>
> Current state → `doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md` · Timeline → `meta-log/` · Kept for history only.
> Archived: 2026-06-28

---

# Goal: intel-qwen36 Qwen3.6 35B A3B Q4_K_M Engine

Status: active
Created: 2026-06-26

## Objective

Create a `meta-engine-factory` style specialized inference engine that runs
only `Qwen3.6-35B-A3B-GGUF` Q4_K_M on the Intel PTL target at the highest
achievable single-request speed. Batch size is locked to `1`.

The engine is valuable only if it is correct under the teacher-forced
acceptance ladder and clears an owner-approved target derived from same-host
floor plus measured roofline. A single short-context microbenchmark is not
acceptance.

## Acceptance Shape

The promoted engine must eventually record:

- exact model path, byte size, and SHA-256
- exact host/kernel/driver/runtime versions
- cold no-prefix matrix from `1k` through `256k` input and `512`/`1k` output
- TTFT, prefill tok/s, TPOT, decode tok/s, total latency, memory state, and
  kernel/submit accounting
- fixed prompt/tokenizer sanity, component numeric checks, teacher-forced
  distribution checks, deterministic token equivalence, and long-context
  sentinel correctness
- long-context smoothness across the matrix
- proof that prefix-cache contamination is not counted as cold prefill
- proof that both prefill and decode clear the accepted target, or an explicit
  route rejection

## Current Gate

Current selector gate:

```text
r1_native_gguf_correctness_first_token_loop
```

R0 setup is closed for target/model facts, oracle bundle validation, route
feasibility, and the current resident harness load path. No engine performance
work is promoted until native output passes oracle-backed correctness checks
and benchmark artifacts satisfy the acceptance matrix.

Latest gate audit:
`output/r1-native-correctness-gate-20260627T062540Z/` records
`r0_ready=true`, `required_checks_passed=true`,
`r1_native_correctness_gate_closed=false`, and
`missing_for_gate=["native_candidate_jsonl"]`. Oracle fixtures, llama.cpp rows,
and OpenVINO rows are not accepted as native correctness evidence.

Latest native model-load prerequisite:
`output/r1-native-gguf-load-map-20260627T063529Z/` records a ready native GGUF
load map for the locked target model: 693 tensors, 30 linear/SSM layers, and 10
full-attention layers. This is not token correctness evidence and keeps
`r1_native_correctness_gate_closed=false`.

Latest engine-side model inspection:
`output/r1-engine-gguf-inspect-20260627T065316Z/` builds and runs the C++
engine GGUF inspector on `ptl-cls-dvt2-008` against the locked model. It
validates the same tensor/layer map from engine code and decodes
representative F32, Q4_K, and Q6_K payload blocks, but still does not run
inference or produce native candidate rows.

Latest engine-side embedding boundary compare:
`output/r1-engine-embedding-compare-20260627T070555Z/` builds and runs the C++
embedding compare on `ptl-cls-dvt2-008`. It decodes `token_embd.weight` row
token id 30 from the locked Q4_K GGUF and exactly matches the oracle
`model.input_embed` payload for `short_math_001` token position 15
(`max_abs_diff=0`, `rmse=0`, `cosine=1`). This is boundary evidence only and
keeps `r1_native_correctness_gate_closed=false`.

Latest engine-side RMSNorm boundary compare:
`output/r1-engine-rmsnorm-compare-20260627T071636Z/` builds and runs the C++
L0 RMSNorm compare on `ptl-cls-dvt2-008`. It decodes the locked embedding row,
reads `blk.0.attn_norm.weight`, applies GGUF epsilon
`9.999999974752427e-07`, and matches the oracle `attn_norm-0` payload within
component numeric thresholds (`max_abs_diff=1.71661376953125e-05`,
`rmse=8.547827141800412e-07`, `cosine=0.9999999999999958`). This is still not
a native candidate token row and keeps `r1_native_correctness_gate_closed=false`.

Latest engine-side QKV boundary compare:
`output/r1-engine-qkv-compare-20260627T073405Z/` builds and runs the C++ L0
qkv compare on `ptl-cls-dvt2-008`. It quantizes the `attn_norm-0` activation
to Q8_K blocks, computes `blk.0.attn_qkv.weight` as Q6_K x Q8_K, and matches
the oracle `linear_attn_qkv_mixed-0` payload within component numeric
thresholds (`max_abs_diff=3.814697265625e-06`,
`rmse=1.9733536165607048e-07`, `cosine=0.9999999999999886`). This is still not
a native candidate token row and keeps `r1_native_correctness_gate_closed=false`.

Latest engine-side attention output projection compare:
`output/r1-engine-attn-output-compare-20260627T074716Z/` builds and runs the
C++ L0 attention output projection compare on `ptl-cls-dvt2-008`. It quantizes
the `final_output-0` activation to Q8_K blocks, computes
`blk.0.ssm_out.weight` as Q4_K x Q8_K, and matches the oracle
`linear_attn_out-0` payload within component numeric thresholds
(`max_abs_diff=8.940696716308594e-08`,
`rmse=1.4396264968756856e-08`, `cosine=0.9999999999991703`). This is still not
a native candidate token row and keeps `r1_native_correctness_gate_closed=false`.

Latest engine-side post-attention residual compare:
`output/r1-engine-attn-residual-compare-20260627T075849Z/` builds and runs the
C++ L0 post-attention residual compare on `ptl-cls-dvt2-008`. It computes
`model.input_embed + linear_attn_out-0` and exactly matches the oracle
`attn_residual-0` payload (`max_abs_diff=0`, `rmse=0`,
`cosine=0.9999999999999999`). This is still not a native candidate token row
and keeps `r1_native_correctness_gate_closed=false`.

Latest engine-side FFN RMSNorm boundary compare:
`output/r1-engine-ffn-rmsnorm-compare-20260627T080720Z/` builds and runs the
C++ L0 FFN RMSNorm compare on `ptl-cls-dvt2-008`. It applies
`blk.0.post_attention_norm.weight` to the oracle `attn_residual-0` payload
with GGUF epsilon `9.999999974752427e-07` and matches the oracle
`attn_post_norm-0` payload within component numeric thresholds
(`max_abs_diff=9.5367431640625e-07`, `rmse=6.498965600081644e-08`,
`cosine=0.9999999999999963`). This is still not a native candidate token row
and keeps `r1_native_correctness_gate_closed=false`.

Latest engine-side router logits boundary compare:
`output/r1-engine-router-logits-compare-20260627T081845Z/` builds and runs the
C++ L0 router logits compare on `ptl-cls-dvt2-008`. It computes
`blk.0.ffn_gate_inp.weight` from the oracle `attn_post_norm-0` payload and
matches the oracle `ffn_moe_logits-0` payload within component numeric
thresholds (`max_abs_diff=1.33514404296875e-05`,
`rmse=4.337466695052883e-06`, `cosine=0.9999999999997536`). This is still not
a native candidate token row and keeps `r1_native_correctness_gate_closed=false`.

Latest engine-side router top-k boundary compare:
`output/r1-engine-router-topk-compare-20260627T082911Z/` builds and runs the
C++ L0 router top-k compare on `ptl-cls-dvt2-008`. It recomputes router logits
from `attn_post_norm-0`, matches expert ids
`[197,196,101,216,105,249,154,104]` exactly, and matches router weights plus
normalized weights within component numeric thresholds
(`weights_max_abs_diff=1.7136335372924805e-07`,
`weights_norm_max_abs_diff=5.364418029785156e-07`). This is still not a native
candidate token row and keeps `r1_native_correctness_gate_closed=false`.

Latest engine-side selected expert gate/up boundary compare:
`output/r1-engine-selected-expert-gate-up-compare-20260627T084406Z/` builds
and runs the C++ L0 selected expert gate/up compare on `ptl-cls-dvt2-008`. It
uses expert ids `[197,196,101,216,105,249,154,104]`, computes
`blk.0.ffn_gate_up_exps.weight` as selected-expert Q4_K x Q8_K matvecs, and
matches the oracle `ffn_moe_gate_up-0` payload within component numeric
thresholds (`max_abs_diff=9.5367431640625e-07`,
`rmse=5.7349144070504335e-08`, `cosine=0.9999999999999865`). This is still
not a native candidate token row and keeps
`r1_native_correctness_gate_closed=false`.

Latest engine-side SWIGLU boundary compare:
`output/r1-engine-swiglu-compare-20260627T085722Z/` builds and runs the C++
L0 SWIGLU compare on `ptl-cls-dvt2-008`. It recomputes the selected expert
gate/up activation for experts `[197,196,101,216,105,249,154,104]`, applies
`gate * sigmoid(gate) * up`, and matches the oracle `ffn_moe_swiglu-0`
payload within component numeric thresholds
(`max_abs_diff=1.1920928955078125e-07`,
`rmse=7.786244864932452e-09`, `cosine=0.999999999999961`). This is still not
a native candidate token row and keeps
`r1_native_correctness_gate_closed=false`.

Latest engine-side selected expert down boundary compare:
`output/r1-engine-selected-expert-down-compare-20260627T090922Z/` builds and
runs the C++ L0 selected expert down compare on `ptl-cls-dvt2-008`. It uses
the oracle `ffn_moe_swiglu-0` activation for experts
`[197,196,101,216,105,249,154,104]`, computes
`blk.0.ffn_down_exps.weight` as selected-expert Q6_K x Q8_K matvecs, and
matches the oracle `ffn_moe_down-0` payload within component numeric
thresholds (`max_abs_diff=7.450580596923828e-09`,
`rmse=8.451740140123001e-10`, `cosine=0.9999999999999789`). This is still
not a native candidate token row and keeps
`r1_native_correctness_gate_closed=false`.

Latest engine-side shared expert boundary compare:
`output/r1-engine-shared-expert-compare-20260627T092259Z/` builds and runs the
C++ L0 shared expert compare on `ptl-cls-dvt2-008`. It uses the oracle
`attn_post_norm-0` activation, computes the shared expert
`blk.0.ffn_gate_shexp.weight`, `blk.0.ffn_up_shexp.weight`, SwiGLU, and
`blk.0.ffn_down_shexp.weight` path, and matches the oracle `ffn_shexp-0`
payload within component numeric thresholds
(`max_abs_diff=1.4901161193847656e-08`,
`rmse=3.087093312454213e-09`, `cosine=0.9999999999999949`). This is still
not a native candidate token row and keeps
`r1_native_correctness_gate_closed=false`.

Latest engine-side MoE residual boundary compare:
`output/r1-engine-moe-residual-compare-20260627T093715Z/` builds and runs the
C++ L0 MoE residual compare on `ptl-cls-dvt2-008`. It verifies normalized
expert weighting, shared expert gating through
`blk.0.ffn_gate_inp_shexp.weight`, the composed `ffn_out-0`, and the derived
`attn_residual-0 + ffn_out-0 -> moe_residual-0` output within component
numeric thresholds (`ffn_out_max_abs_diff=2.682209014892578e-07`,
`moe_residual_max_abs_diff=2.682209014892578e-07`,
`moe_residual_rmse=7.760108889930047e-09`,
`moe_residual_cosine=0.9999999999998894`). This is still not a native
candidate token row and keeps `r1_native_correctness_gate_closed=false`.

Latest engine-side final norm boundary compare:
`output/r1-engine-final-norm-compare-20260627T094922Z/` builds and runs the
C++ global `output_norm.weight` RMSNorm compare on `ptl-cls-dvt2-008`. It
applies final norm to the oracle `l_out-39__tok15__ord1490.bin` payload and
compares against `result_norm__tok15__ord1491.bin` within component numeric
thresholds (`max_abs_diff=1.1444091796875e-05`,
`rmse=8.575131329283268e-07`, `cosine=0.9999999999999962`). This is still not
a native candidate token row and keeps `r1_native_correctness_gate_closed=false`.

Latest engine-side LM head boundary compare:
`output/r1-engine-lm-head-compare-20260627T100103Z/` builds and runs the C++
global `output.weight` matvec compare on `ptl-cls-dvt2-008`. It applies the
Q6_K LM head to the oracle `result_norm__tok15__ord1491.bin` payload and
compares against `result_output__tok15__ord1492.bin` logits within component
numeric thresholds (`max_abs_diff=2.86102294921875e-06`,
`rmse=4.318970378914945e-07`, `cosine=0.9999999999998903`). This is still not
a native candidate token row and keeps `r1_native_correctness_gate_closed=false`.

Latest engine-side sampler boundary compare:
`output/r1-engine-sampler-compare-20260627T100947Z/` builds and runs the C++
deterministic top-k sampler compare on `ptl-cls-dvt2-008`. It recomputes the
top-8 rows from `result_output__tok15__ord1492.bin` and matches
`sampler-topk.json` token ids exactly with top token id `271`; max logit diff
is `4.76837158203125e-05`, reflecting JSON write precision. This is still not
a native candidate token row and keeps `r1_native_correctness_gate_closed=false`.

Latest R1 native candidate route artifact:
`output/r1-native-candidate-route-20260627T144754Z/` records the next
promotion route after the component ladder reached the global sampler and the
L0 stateful linear-attention layer path, the L1 post-conv layer core, and the
L3 full-attention q/k/v projection, RoPE, captured-history attention core,
stateful K/V append, gate, and output projection boundaries for
`short_math_001` token position 15. It validates six oracle seed row ids, 29
registered engine-side component
compare artifacts, the open R1 native gate, the native GGUF load map, and the
engine GGUF inspect artifact. It selects
`assemble_o1_first_token_native_loop_from_verified_components` as the next
route, but emits no candidate JSONL, keeps
`r1_native_correctness_gate_closed=false`, and keeps speedup claims forbidden.
The remaining required artifact is a real `intel_qwen36_native`
`native_candidate_jsonl` for all six short/router seed rows.

Latest engine-side FFN block compare:
`output/r1-engine-ffn-block-compare-20260627T103048Z/` builds and runs the C++
parameterized L0 FFN/MoE block on `ptl-cls-dvt2-008`. It starts from oracle
`attn_residual-0`, computes FFN RMSNorm, router logits/top-k, selected expert
gate/up, SwiGLU, selected expert down, normalized expert weighting, shared
expert, `ffn_out`, and derived `moe_residual`. Native top-k exactly matches
`[197,196,101,216,105,249,154,104]`; `ffn_out` max abs diff is
`2.4586915969848633e-07`, and `moe_residual` max abs diff is
`2.384185791015625e-07`. This is a parameterized sublayer promotion, not a
full model loop or native candidate token row, and keeps
`r1_native_correctness_gate_closed=false`.

Latest engine-side layer shell compare:
`output/r1-engine-layer-shell-compare-20260627T104336Z/` builds and runs a
C++ L0 layer shell on `ptl-cls-dvt2-008`. It starts from oracle
`model.input_embed` plus an external `final_output-0` attention state, applies
the engine output projection through `blk.0.ssm_out.weight`, computes the
post-attention residual, then runs the parameterized FFN/MoE sublayer through
derived `moe_residual`. Native top-k exactly matches
`[197,196,101,216,105,249,154,104]`; attention output max abs diff is
`8.940696716308594e-08`, attention residual max abs diff is
`8.195638656616211e-08`, `ffn_out` max abs diff is
`2.682209014892578e-07`, and layer output max abs diff is
`2.086162567138672e-07`. This validates projection/residual/FFN plumbing from
an external attention state; it does not implement the missing attention/SSM
state update, does not emit native candidate rows, and keeps
`r1_native_correctness_gate_closed=false`.

Latest engine-side loop shell compare:
`output/r1-engine-loop-shell-compare-20260627T110740Z/` builds and runs the
C++ 40-layer loop shell on `ptl-cls-dvt2-008` in
`teacher_forced_oracle` residual mode. It reuses one parameterized layer for
all 40 layers, consumes oracle residual and attention projection inputs,
applies each layer's projection/residual/FFN path, then runs final norm, LM
head, and deterministic sampler top-k. The run has top-k mismatch total 0,
sampler token mismatch 0, max attention output diff
`5.960464477539062e-07`, max attention residual diff
`4.76837158203125e-07`, max layer output diff
`0.0005267579108476639`, final norm RMSE
`1.5700917353995573e-06`, logits RMSE
`8.579865841243006e-07`, and top token id `271`. This validates O(1)
layer-shell reuse through the global sampler, but it still does not implement
attention/SSM or KV state updates, does not emit native candidate JSONL rows,
and keeps `r1_native_correctness_gate_closed=false`.

Latest engine-side linear attention delta compare:
`output/r1-engine-linear-attn-delta-compare-20260627T112938Z/` builds and runs
the C++ L0 fused gated-delta core on `ptl-cls-dvt2-008`. It consumes oracle
predelta `q`, `k`, `v`, scalar `gate`, `beta_sigmoid`, `state_predelta`, and
`z` payloads, decodes `blk.0.ssm_norm.weight`, updates the recurrent state,
computes `attn_output`, and applies the gated RMSNorm to produce
`final_output`. The attention output comparison has max abs diff
`1.1920928955078125e-07`, RMSE `4.099912552342994e-09`, cosine
`0.9999999999999947`, and mismatch count 0. The final output comparison has
max abs diff `5.960464477539062e-07`, RMSE
`9.369065450970646e-09`, cosine `1.000000000000001`, and mismatch count 0.
This validates the predelta delta-core math only; it still does not implement
the convolution/state input path, full-attention KV updates, native candidate
JSONL rows, or speedup claims, and keeps
`r1_native_correctness_gate_closed=false`.

Latest engine-side linear attention pre-conv compare:
`output/r1-engine-linear-attn-preconv-compare-20260627T121529Z/` builds and
runs the C++ L0 pre-conv projection path from oracle `attn_norm` on
`ptl-cls-dvt2-008`. It computes `linear_attn_qkv_mixed`, `ssm_alpha`,
`ssm_beta`, softplus/sigmoid gate values, and `z`. It matches
`linear_attn_qkv_mixed` with max abs diff `3.814697265625e-06`, `alpha` and
`beta` with max abs diff `9.5367431640625e-07`, and `z` with max abs diff
`1.9073486328125e-06`. This still does not implement full-attention KV
updates, native candidate JSONL rows, or speedup claims, and keeps
`r1_native_correctness_gate_closed=false`.

Latest engine-side linear attention conv-state compare:
`output/r1-engine-linear-attn-conv-compare-20260627T123037Z/` builds and runs
the C++ L0 recurrent convolution state path on `ptl-cls-dvt2-008`. It starts
from the `short_math_001` prompt token ids, decodes embeddings, computes
RMSNorm and pre-conv projections for tokens 0 through 15, maintains the
3-token convolution history, and compares token 15 against the oracle
payloads. `model_input_embed` matches exactly, `attention_norm` max abs diff
is `1.71661376953125e-05`, `linear_attn_qkv_mixed` max abs diff is
`2.288818359375e-05`, and `conv_output_raw` max abs diff is
`1.811981201171875e-05`. This validates the L0 recurrent convolution
state input/update path only; it still does not implement full-attention KV
updates, native candidate JSONL rows, or speedup claims, and keeps
`r1_native_correctness_gate_closed=false`.

Latest engine-side linear attention post-conv compare:
`output/r1-engine-linear-attn-postconv-compare-20260627T114940Z/` builds and
runs the C++ L0 linear attention path from oracle `conv_output_raw` on
`ptl-cls-dvt2-008`. It computes `ssm_alpha`, `ssm_beta`, `attn_gate`, SILU,
Q/K/V split, Q/K L2 normalization, gated-delta core, gated RMSNorm, and
`ssm_out` projection. It matches `alpha` and `beta` with max abs diff
`9.5367431640625e-07`, `z` with max abs diff
`1.9073486328125e-06`, `q_conv_predelta` and `k_conv_predelta` with max abs
diff `5.960464477539063e-08`, `final_output` with max abs diff
`5.960464477539062e-07`, and `linear_attn_out` with max abs diff
`1.043081283569336e-07`. This still does not implement the recurrent
convolution state input path, full-attention KV updates, native candidate
JSONL rows, or speedup claims, and keeps
`r1_native_correctness_gate_closed=false`.

Latest engine-side layer post-conv compare:
`output/r1-engine-layer-postconv-compare-20260627T120307Z/` builds and runs a
C++ L0 layer path on `ptl-cls-dvt2-008` from native attention norm through
linear attention post-conv, `ssm_out`, post-attention residual, and FFN/MoE.
It consumes oracle `conv_output_raw` and `state_predelta`, so it still is not
a full native layer state-update path. Native top-k exactly matches
`[197,196,101,216,105,249,154,104]`; `final_output` max abs diff is
`4.6193599700927734e-07`, `linear_attention_out` max abs diff is
`9.685754776000977e-08`, attention residual max abs diff is
`9.685754776000977e-08`, and layer output max abs diff is
`4.178308881819248e-05`. This does not emit native candidate JSONL rows, close
R1 token correctness, or allow speed claims, and keeps
`r1_native_correctness_gate_closed=false`.

Latest engine-side stateful linear-attention layer compare:
`output/r1-engine-layer-stateful-linear-attn-compare-20260627T124454Z/`
builds and runs the C++ L0 layer path on `ptl-cls-dvt2-008` while replaying
the 16-token `short_math_001` prompt through engine-maintained convolution and
gated-delta recurrent state. Token 15 matches oracle payloads with
`state_predelta` max abs diff `1.6689300537109375e-05`, `final_output` max
abs diff `4.6193599700927734e-07`, `linear_attention_out` max abs diff
`1.1920928955078125e-07`, and `layer_output` max abs diff
`1.4901161193847656e-07`; router top-k exactly matches
`[197,196,101,216,105,249,154,104]`. This removes oracle
`conv_output_raw`/`state_predelta` dependence for the L0 linear-attention
layer, but still does not implement the remaining 29 linear layers,
full-attention KV updates, native candidate JSONL rows, or speed claims, and
keeps `r1_native_correctness_gate_closed=false`.

Latest engine-side L1 layer post-conv compare:
`output/r1-engine-layer1-postconv-compare-20260627T130117Z/` builds and runs
the parameterized C++ layer core for layer 1 on `ptl-cls-dvt2-008` from oracle
`l_out-0`, `conv_output_raw-1`, and `state_predelta-1`. Token 15 matches
oracle payloads with `final_output` max abs diff
`7.078051567077637e-08`, `linear_attention_out` max abs diff
`3.725290298461914e-08`, and `layer_output` max abs diff
`2.7008354663848877e-08`; router top-k exactly matches
`[205,75,4,97,27,11,41,123]`. This promotes the O(1) layer core to an L1
boundary, but still does not replay engine-maintained L1 state from prior
tokens, implement full-attention KV updates, emit native candidate JSONL rows,
or allow speed claims.

Latest engine-side L3 full-attention q/k/v projection compare:
`output/r1-engine-full-attn-qkv-compare-20260627T131936Z/` builds and runs
the C++ first full-attention projection boundary on `ptl-cls-dvt2-008` from
oracle `l_out-2` for `short_math_001` token 15. It validates `attn_norm-3`,
`Qcur_full-3`, `Qcur_normed-3`, `Kcur-3`, `Kcur_normed-3`, and `Vcur-3` with
max abs diffs `7.62939453125e-06`, `3.814697265625e-06`,
`2.86102294921875e-06`, `1.430511474609375e-06`,
`2.1457672119140625e-06`, and `8.344650268554688e-07`. This establishes the
interleaved Q/gate split for `blk.3.attn_q.weight` plus per-head Q/K RMSNorm,
but still does not apply RoPE, update the full-attention KV cache, compute
attention output, emit native candidate JSONL rows, or allow speed claims.

Latest engine-side L3 full-attention RoPE compare:
`output/r1-engine-full-attn-rope-compare-20260627T134454Z/` builds and runs
the C++ first full-attention IMRoPE boundary on `ptl-cls-dvt2-008` from
oracle `Qcur_normed-3` and `Kcur_normed-3` for `short_math_001` token 15.
It validates `rope_dimension_count=64`, sections `[11,11,10,0]`, text
position ids `[15,15,15,0]`, and `rope_freq_base=10000000`. The `q_rope`
comparison has max abs diff `4.76837158203125e-07`; `k_rope` max abs diff is
`2.384185791015625e-07`, with mismatch count 0 for both. This validates RoPE
only; it still does not update the full-attention KV cache, compute attention
output, emit native candidate JSONL rows, or allow speed claims.

Latest L3 full-attention history capture:
`output/r1-full-attn-history-capture-20260627T142546Z/` captures
`short_math_001` token positions 0 through 15 on `ptl-cls-dvt2-008` using the
patched boundary-capture executable. It retains the layer-3 `q_rope`,
`k_rope`, `v`, `attn_pregate`, `attn_gated`, and `attn_output` payloads for
each token. This artifact exists to supply historical K/V to the attention
core compare; it is not native candidate JSONL evidence.

Latest engine-side L3 full-attention core compare:
`output/r1-engine-full-attn-core-compare-20260627T143426Z/` builds and runs
the C++ causal full-attention core compare on `ptl-cls-dvt2-008` using the
captured token 0..15 L3 K/V history. It validates `head_dim=256`,
`q_head_count=16`, `kv_head_count=2`, `gqa_group=8`, and attention scale
`0.0625`, then compares native `attn_pregate` against oracle
`attn_pregate-3` for token 15. The selected `f32_source_payload` mode has max
abs diff `0.0017899274826049805`, RMSE `0.00018489847371300422`, cosine
`0.99999986970357`, and mismatch count 0. This validates attention math from
captured history, but still does not validate native KV-cache updates, emit
native candidate JSONL rows, or allow speed claims.

Latest engine-side L3 stateful full-attention layer compare:
`output/r1-engine-full-attn-stateful-layer-compare-20260627T144639Z/` builds
and runs the C++ stateful full-attention layer compare on `ptl-cls-dvt2-008`.
It starts from oracle `l_out-2`, computes token15 Q/K/V/RoPE natively,
appends native K/V to captured token0..14 history, runs causal attention,
applies the Q gate, and projects through `blk.3.attn_output.weight`. Appended
K matches oracle `Kcur-3` with max abs diff `2.1457672119140625e-06`;
appended V matches oracle `Vcur-3` with max abs diff
`8.344650268554688e-07`; pregate max abs diff is
`0.0017900466918945312`; final `attn_output-3` max abs diff is
`0.00007873540744185448`, RMSE `0.000012387206664931045`, cosine
`0.9999985338100779`, and mismatch count 0. This validates one token's
native full-attention K/V append plus attention/gate/output path, but still
does not generate all full-attention histories, emit native candidate JSONL
rows, or allow speed claims.

Latest all-layer full-attention history capture:
`output/r1-full-attn-all-history-capture-20260627T145615Z/` captures
`short_math_001` token positions 0 through 15 on `ptl-cls-dvt2-008` for all
10 full-attention layers `[3,7,11,15,19,23,27,31,35,39]`. It retains each
layer's `q_rope`, `k_rope`, `v`, `attn_pregate`, `attn_gated`, and
`attn_output` payloads, for 960 selected payloads total. This is oracle
capture evidence for multi-layer full-attention KV-update validation, not
native candidate JSONL evidence.

Latest engine-side all-layer stateful full-attention compare:
`output/r1-engine-full-attn-all-stateful-layers-compare-20260627T151140Z/`
builds and runs the C++ all-layer stateful full-attention compare on
`ptl-cls-dvt2-008`. It stages 390 payloads from the all-layer history capture
and R0 token15 dump, then validates token15 native K/V append plus causal
attention, gate, and output projection for layers
`[3,7,11,15,19,23,27,31,35,39]`. All 10 layers pass with component numeric
thresholds max abs `0.0125`, RMSE `0.001`, cosine `0.99998`; worst final
`attn_output` max abs diff is `0.0028264522552490234`, RMSE is
`0.00039453445936383765`, and cosine is `0.9999805655227852`. This still
consumes captured token0..14 K/V history and oracle layer inputs, does not
emit native candidate JSONL rows, and does not allow speed claims.

Latest engine-side all-linear post-conv compare:
`output/r1-engine-linear-attn-all-postconv-compare-20260627T153914Z/`
builds and runs the C++ all-linear post-conv compare on `ptl-cls-dvt2-008`.
It stages 540 token15 payloads from the R0 tensor dump and validates
linear-attention post-conv math for all 30 linear layers using oracle
`conv_output_raw` and `state_predelta` inputs. All layers pass strict
component thresholds max abs `0.0005`, RMSE `0.00005`, cosine `0.99999`.
Worst `attention_output` max abs diff is `1.1920928955078125e-07`, worst
`final_output` max abs diff is `5.960464477539062e-07`, and worst
`linear_attn_out` max abs diff is `0.00010106712579727173`. This does not
prove convolution history replay, residual chaining, the integrated 40-layer
loop, native candidate JSONL rows, or speed claims.

Latest engine-side seed prompt input check:
`output/r1-engine-seed-prompt-input-check-20260627T155328Z/` builds and runs
the C++ seed prompt input checker on `ptl-cls-dvt2-008`. It materializes the
six oracle seed rows as `cases.tsv` plus little-endian u32 token files,
validates 150 total prompt tokens, 98 unique token IDs/embedding rows, and
the per-case FNV64 signatures
`3b6f72e30ebbd065`, `ecfa12cde35a9b08`, `1dcccdcb6b408515`,
`24e3daa3b2d1cf38`, `7825f48fab46c02c`, and `1970eff24e856c27`. It also
replays the `short_math_001` final prompt token embedding exactly against the
R0 oracle payload with max abs diff 0, RMSE 0, cosine 1. This closes the
native seed prompt token input path prerequisite only; it does not emit native
candidate JSONL rows or allow speed claims.

Latest R1 native candidate route:
`output/r1-native-candidate-route-20260627T155549Z/` registers 31 component
compare artifacts, including the all-30-linear token15 post-conv compare and
the all-10-full-attention stateful K/V append compare, plus 1 prerequisite
evidence artifact for the native seed prompt input path. The frontier is
`global_sampler_topk_plus_l0_stateful_linear_attention_layer_plus_l1_postconv_core_plus_all_30_linear_attention_postconv_core_plus_l3_full_attention_qkv_projection_plus_l3_full_attention_rope_plus_l3_full_attention_core_from_captured_kv_history_plus_l3_stateful_full_attention_kv_append_gate_output_projection_plus_all_10_full_attention_stateful_kv_append_gate_output_projection_for_short_math_001_tok15`.
R1 remains open: the next required artifact is still a real
`intel_qwen36_native` candidate JSONL for all six seed rows.

Latest engine-side L3 full-attention gate compare:
`output/r1-engine-full-attn-gate-compare-20260627T141107Z/` builds and runs
the C++ first full-attention gate boundary on `ptl-cls-dvt2-008` from oracle
`Qcur_full-3` and oracle `attn_pregate-3` for `short_math_001` token 15. It
validates the per-head `[query(256), gate(256)]` split, applies
`sigmoid(gate) * attn_pregate`, and compares against oracle `attn_gated-3`.
The 4096-value output has max abs diff `3.725290298461914e-09`, RMSE
`2.558733662346639e-10`, cosine `0.9999999999999973`, and mismatch count 0.
This validates the gate fed by oracle pregate attention state only; it still
does not update the full-attention KV cache, compute attention, emit native
candidate JSONL rows, or allow speed claims.

Latest engine-side L3 full-attention output projection compare:
`output/r1-engine-full-attn-output-projection-compare-20260627T135544Z/`
builds and runs the C++ first full-attention output projection boundary on
`ptl-cls-dvt2-008` from oracle `attn_gated-3` for `short_math_001` token 15.
It computes `blk.3.attn_output.weight` and compares against oracle
`attn_output-3`. The 2048-value output has max abs diff
`2.2351741790771484e-08`, RMSE `2.0953007404094846e-09`, cosine
`0.9999999999999576`, and mismatch count 0. This validates the projection fed
by oracle attention state only; it still does not update the full-attention KV
cache, compute attention, emit native candidate JSONL rows, or allow speed
claims.

Diagnostic two-linear-layer native replay:
`output/r1-engine-two-linear-layers-stateful-compare-20260627T125708Z/`
attempted to replay L0 and L1 linear-attention layers with native state
handoff. It kept router top-k exact for layer 1, but downstream L1 outputs
exceeded the component relative-L2 threshold, so it was not registered as an
accepted compare artifact.

## Non-Goals

- generic OpenAI-compatible serving
- multi-model loading
- dynamic batching
- optimizing for batch size greater than `1`
- changing target firmware, kernel, or system packages without an explicit
  rollback plan
