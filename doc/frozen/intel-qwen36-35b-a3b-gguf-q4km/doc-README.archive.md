> **ARCHIVED — DO NOT EDIT.** Frozen snapshot of append-only progress narration,
> moved out of the live docs during the 2026-06-28 documentation distillation.
>
> Current state → `doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md` · Timeline → `meta-log/` · Kept for history only.
> Archived: 2026-06-28

---

# intel-qwen36 / doc Index

Snapshot: 2026-06-26

Stable conclusions, plans, references, SOPs, decisions, and ledgers belong in
`doc/`. Process history belongs in `meta-log/`.

## Current Entry

Read these in order for a new session:

1. `.meta-agent/AGENT-RUNTIME.md`
2. `AGENTS.md`
3. `meta-engine-factory/doc/methodology/00-engine-and-roadmap.md`
4. `meta-engine-factory/doc/operationalization/day-0-bootstrap.md`
5. `meta-engine-factory/doc/operationalization/resident-harness.md`
6. `doc/WORKSTREAMS.md`
7. `goals/intel-qwen36-35b-a3b-q4km-engine.md`
8. `contracts/intel-qwen36-target-contract.json`
9. `contracts/qwen36-35b-a3b-gguf-q4km-model-contract.json`
10. `benchmarks/intel-qwen36-35b-a3b-gguf-q4km/acceptance-matrix.json`
11. `benchmarks/intel-qwen36-35b-a3b-gguf-q4km/prompt-suites.json`
12. `oracle/oracle-bundle-contract.json`
13. `doc/active/intel-qwen36-35b-a3b-gguf-q4km/intel-qwen36-35b-a3b-gguf-q4km-day0-r0-plan-2026-06-26.md`

## Mission

Build the fastest correct locked batch-size-1 native engine for
`Qwen3.6-35B-A3B-GGUF` Q4_K_M on the Intel PTL target.

## Current State

The repository is day-0 scaffolded. It has:

- `meta-agent` submodule for agent workflow
- `meta-engine-factory` submodule for engine methodology
- locked target and model contracts
- acceptance matrix with R0 refresh requirement
- prompt-suite specs for oracle capture and denominator runs
- oracle bundle contract with a validated full R0 oracle bundle
- performance seed audit with prior denominator/roofline diagnostics
- current-target denominator preflight through 262144 prompt materialization
- OpenVINO 262144 denominator diagnostic rejected on GPU
  `CL_OUT_OF_RESOURCES` even at `-mt 1`
- current-target native GGUF source-stream roof rejected at 48.51-64.20 GB/s
  against the 115 GB/s route line
- current-target real-tensor OpenCL qmatvec numeric check passed, but default
  kernels reached only 25.81-26.12 effective tensor GB/s
- per-bucket KV/read pressure estimate shows fp16/bf16 256K decode needs
  5.37 GB of full-attention KV reads per output token before active weights
- route-feasibility decision hard-rejected the raw OpenCL GGUF
  source-stream/qmatvec route and selected denominator/oracle closure next
- denominator/oracle boundary resolution records that the OpenVINO 262144
  resource failure is unavailable-lane evidence, not a throughput denominator
- llama.cpp/Vulkan denominator route preflight passed and produced 128/4096
  smoke rows, but the 262144 paired run timed out without a metric after the
  long run was cleaned up
- R0 policy accepts the 262144 denominator lane as unavailable; this closes
  only denominator interpretation and still forbids 262144 speedup claims
- full oracle capture spec exists for 17 boundary types, 520 per-layer boundary
  records, and the 26-row prompt ladder; it is not a captured bundle
- oracle runtime preflight found a usable llama.cpp prior-tool route for
  distribution capture via `completion_probabilities`; stock llama.cpp/OpenVINO
  paths still do not expose the required per-boundary tensor bundle
- boundary capture route preflight confirms the target llama.cpp runtime is an
  installed binary-only tree with no instrumentable source tree, but the Intel
  env user-space `cmake`/`g++`/`ninja` toolchain is available; the target
  runtime reports `version: 9518 (7c158fbb4)`
- llama.cpp source build route resolved build 9518 to official upstream commit
  `7c158fbb4aec1bdc9c81d6ca0e785139f4826fae` and staged that source tree under
  `/home/intel/intel-qwen36-r0/source/` for instrumentation
- current-target distribution capture smoke produced a real
  `completion_probabilities` row for `short_math_001` with one generated token
  and top-5 logprobs; it is smoke evidence only, not a full distribution bundle
- current-target short/router distribution capture produced six
  `completion_probabilities` rows with 429 total top-5 logprob positions; two
  short cases stopped at EOS before their request limit, and the artifact is
  still only a short/router subset
- current-target materialized-prompt distribution capture covers 1k through 128k
  sentinel/prefill rows with 7,154 total top-5 logprob positions; this is still
  only a materialized subset, not the full distribution bundle
- separate 1024-token-request materialized-prompt distribution captures cover
  1k through 128k sentinel/prefill rows with 12,315 total top-5 logprob
  positions; this is still a materialized subset, not the full distribution
  bundle
- full oracle capture queue expands the spec into 1100 required bundle JSONL
  rows: 26 token/top-k tasks, 26 distribution tasks, 524 boundary input tensor
  tasks, and 524 boundary output tensor tasks
- full oracle prompt materialization now exists for all 26 prompt rows; the 20
  generated long-context and prefill rows have exact current-target
  llama.cpp-tokenizer counts through 262144 tokens
- full-ladder prompt token IDs are captured for all 26 oracle rows, totaling
  1,251,478 prompt tokens; this still lacks full-ladder top-k logits
- bounded first-token top-k smoke now covers materialized 1k through 128k
  sentinel/prefill rows, plus one short row, with top-5 logprobs; it is route
  evidence only, not the full-ladder top-k bundle
- exact 262144-token sentinel/prefill top-k smoke was attempted with
  llama.cpp CPU server at `n_ctx=262144`; both rows returned
  `exceed_context_size_error`, so 256k top-k remains unresolved under the
  current context contract
- R0 prompt-edge policy accepts exact 262144-token first-token top-k as a
  context-edge unavailable row under the locked model context contract; this
  does not create logits or close the oracle gate
- full oracle bundle validator now enforces the real closure shape: 524 total
  boundary records, 26 prompt rows, explicit 256k prompt-edge rows, full
  teacher-forced distributions, and per-boundary input/output tensors
- staged llama.cpp build `9518 (7c158fbb4)` source is mapped to all 17 required
  oracle boundary hook points across Qwen35MoE, `llama-graph`, and sampler code;
  this is a patch/build route map only and does not capture tensors
- staged llama.cpp source now has a dedicated
  `llama-qwen36-boundary-capture` executable patched and built under the Intel
  env toolchain; `--help` passes, and the enhanced locked-model
  `short_math_001` source-position run captured 1,493 tensor rows including
  linear-attention tensors
- hybrid-aware boundary capture coverage has effective policy coverage for all
  524 input tasks and all 524 output tasks
- boundary bundle fragment assembly produced `boundary-references/inputs.jsonl`
  and `boundary-references/outputs.jsonl` with 524 rows each, including 60
  policy-not-applicable linear-layer RoPE rows and 40 derived MoE residual
  outputs
- full oracle bundle `oracle/r0-oracle-bundle-20260627T060028Z/` is validated:
  26 token/top-k rows, 26 teacher-forced distribution rows with 12,744
  top-logprob positions, 524 boundary input rows, 524 boundary output rows, and
  explicit 256k prompt-edge rows; latest validation is
  `output/r0-oracle-bundle-validation-20260627T060238Z/` with
  `r0_oracle_gate_closed=true`
- resident harness load artifact
  `output/r0-resident-harness-load-20260627T061911Z/` called
  `build/engine/iq36-load-bundle` with the locked model path and validated
  oracle bundle, returned 0, entered loaded state, and read oracle row counts
  26/26/524/524
- post-load resident harness gate audit
  `output/r0-resident-harness-gate-audit-20260627T061917Z/` records
  `r0_resident_harness_gate_closed=true`
- resident harness skeleton now rejects placeholder or nonexistent oracle
  bundle paths, and requires the real bundle directory layout before `load`
- R1 native correctness gate artifact
  `output/r1-native-correctness-gate-20260627T062540Z/` records
  `r0_ready=true`, requires future rows to declare
  `native_output_source=intel_qwen36_native`, rejects oracle/reference-runtime
  evidence, and keeps `r1_native_correctness_gate_closed=false` until a real
  native candidate JSONL exactly replays the six short/router oracle rows
- R1 native GGUF load-map artifact
  `output/r1-native-gguf-load-map-20260627T063529Z/` parses the locked target
  GGUF tensor table and validates 693 tensors: 301 F32, 331 Q4_K, 61 Q6_K,
  with 30 linear/SSM layers and 10 full-attention layers at indexes
  `3,7,11,15,19,23,27,31,35,39`; this is model-load evidence only and does
  not close native token correctness
- R1 engine-side GGUF inspect artifact
  `output/r1-engine-gguf-inspect-20260627T065316Z/` stages the C++ engine
  parser to `ptl-cls-dvt2-008`, builds `iq36-gguf-inspect` with the Intel env,
  runs it against the locked GGUF, and validates the same 693-tensor 30/10
  layer map from the engine code path; it also decodes representative F32,
  Q4_K, and Q6_K payload blocks from `output_norm.weight`,
  `token_embd.weight`, and `output.weight`; this still does not run inference
  or generate native candidate rows
- R1 engine-side embedding compare artifact
  `output/r1-engine-embedding-compare-20260627T070555Z/` stages the C++
  engine row decoder to `ptl-cls-dvt2-008`, decodes `token_embd.weight` row
  token id 30 from the locked Q4_K GGUF, and compares it with the oracle
  `model.input_embed` payload for `short_math_001` token position 15. The
  2048-value vector matches exactly: max abs diff 0, RMSE 0, cosine 1. This
  still does not run the full model loop or generate native candidate rows
- R1 engine-side RMSNorm compare artifact
  `output/r1-engine-rmsnorm-compare-20260627T071636Z/` stages the C++ RMSNorm
  boundary compare to `ptl-cls-dvt2-008`, decodes the same embedding row plus
  `blk.0.attn_norm.weight`, applies GGUF epsilon
  `9.999999974752427e-07`, and compares against the oracle `attn_norm-0`
  payload. The 2048-value vector passes component numeric thresholds:
  max abs diff `1.71661376953125e-05`, RMSE
  `8.547827141800412e-07`, cosine `0.9999999999999958`
- R1 engine-side QKV compare artifact
  `output/r1-engine-qkv-compare-20260627T073405Z/` stages the C++ GGUF matvec
  compare to `ptl-cls-dvt2-008`, quantizes the `attn_norm-0` activation to
  Q8_K blocks, computes the L0 `blk.0.attn_qkv.weight` Q6_K x Q8_K dot path,
  and compares against the oracle `linear_attn_qkv_mixed-0` payload. The
  8192-value vector passes component numeric thresholds: max abs diff
  `3.814697265625e-06`, RMSE `1.9733536165607048e-07`, cosine
  `0.9999999999999886`, mismatch count 0. This still does not run the full
  model loop or generate native candidate rows
- R1 engine-side attention output projection compare artifact
  `output/r1-engine-attn-output-compare-20260627T074716Z/` stages the C++
  GGUF matvec compare to `ptl-cls-dvt2-008`, quantizes the `final_output-0`
  activation to Q8_K blocks, computes `blk.0.ssm_out.weight` as Q4_K x Q8_K,
  and compares against the oracle `linear_attn_out-0` payload. The
  2048-value vector passes component numeric thresholds: max abs diff
  `8.940696716308594e-08`, RMSE `1.4396264968756856e-08`, cosine
  `0.9999999999991703`, mismatch count 0. This still does not run the full
  model loop or generate native candidate rows
- R1 engine-side post-attention residual compare artifact
  `output/r1-engine-attn-residual-compare-20260627T075849Z/` stages the C++
  residual add compare to `ptl-cls-dvt2-008`, computes
  `model.input_embed + linear_attn_out-0`, and compares against the oracle
  `attn_residual-0` payload. The 2048-value vector matches exactly: max abs
  diff 0, RMSE 0, cosine `0.9999999999999999`, mismatch count 0. This still
  does not run the full model loop or generate native candidate rows
- R1 engine-side FFN RMSNorm compare artifact
  `output/r1-engine-ffn-rmsnorm-compare-20260627T080720Z/` stages the C++
  FFN RMSNorm compare to `ptl-cls-dvt2-008`, applies
  `blk.0.post_attention_norm.weight` to the oracle `attn_residual-0` payload
  with GGUF epsilon `9.999999974752427e-07`, and compares against the oracle
  `attn_post_norm-0` payload. The 2048-value vector passes component numeric
  thresholds: max abs diff `9.5367431640625e-07`, RMSE
  `6.498965600081644e-08`, cosine `0.9999999999999963`, mismatch count 0.
  This still does not run the full model loop or generate native candidate
  rows
- R1 engine-side router logits compare artifact
  `output/r1-engine-router-logits-compare-20260627T081845Z/` stages the C++
  F32 router matvec compare to `ptl-cls-dvt2-008`, computes
  `blk.0.ffn_gate_inp.weight` from the oracle `attn_post_norm-0` payload, and
  compares against the oracle `ffn_moe_logits-0` payload. The 256-value vector
  passes component numeric thresholds: max abs diff
  `1.33514404296875e-05`, RMSE `4.337466695052883e-06`, cosine
  `0.9999999999997536`, mismatch count 0. This still does not run the full
  model loop or generate native candidate rows
- R1 engine-side router top-k compare artifact
  `output/r1-engine-router-topk-compare-20260627T082911Z/` stages the C++
  router softmax/top-k compare to `ptl-cls-dvt2-008`, recomputes the L0 router
  logits from `attn_post_norm-0`, selects experts
  `[197,196,101,216,105,249,154,104]`, and compares router weights plus
  normalized weights against the oracle payloads. Top-k mismatch count is 0;
  weights max abs diff is `1.7136335372924805e-07`; normalized weights max abs
  diff is `5.364418029785156e-07`. This still does not run the full model loop
  or generate native candidate rows
- R1 engine-side selected expert gate/up compare artifact
  `output/r1-engine-selected-expert-gate-up-compare-20260627T084406Z/`
  stages the C++ selected-expert Q4_K x Q8_K matvec compare to
  `ptl-cls-dvt2-008`, uses experts `[197,196,101,216,105,249,154,104]`, and
  compares `blk.0.ffn_gate_up_exps.weight` against the oracle
  `ffn_moe_gate_up-0` payload. The 8192-value vector passes component numeric
  thresholds: max abs diff `9.5367431640625e-07`, RMSE
  `5.7349144070504335e-08`, cosine `0.9999999999999865`, mismatch count 0.
  This still does not run the full model loop or generate native candidate rows
- R1 engine-side SWIGLU compare artifact
  `output/r1-engine-swiglu-compare-20260627T085722Z/` stages the C++ SWIGLU
  compare to `ptl-cls-dvt2-008`, recomputes the selected expert gate/up
  activation for experts `[197,196,101,216,105,249,154,104]`, applies
  `gate * sigmoid(gate) * up`, and compares against the oracle
  `ffn_moe_swiglu-0` payload. The 4096-value vector passes component numeric
  thresholds: max abs diff `1.1920928955078125e-07`, RMSE
  `7.786244864932452e-09`, cosine `0.999999999999961`, mismatch count 0.
  This still does not run the full model loop or generate native candidate rows
- R1 engine-side selected expert down compare artifact
  `output/r1-engine-selected-expert-down-compare-20260627T090922Z/` stages
  the C++ selected-expert down projection compare to `ptl-cls-dvt2-008`, uses
  oracle `ffn_moe_swiglu-0` plus experts `[197,196,101,216,105,249,154,104]`,
  and compares `blk.0.ffn_down_exps.weight` against the oracle
  `ffn_moe_down-0` payload. The Q6_K 16384-value vector passes component
  numeric thresholds: max abs diff `7.450580596923828e-09`, RMSE
  `8.451740140123001e-10`, cosine `0.9999999999999789`, mismatch count 0.
  This still does not run the full model loop or generate native candidate rows
- R1 engine-side shared expert compare artifact
  `output/r1-engine-shared-expert-compare-20260627T092259Z/` stages the C++
  shared expert gate/up/SwiGLU/down compare to `ptl-cls-dvt2-008`, uses the
  oracle `attn_post_norm-0` input, computes `blk.0.ffn_gate_shexp.weight`,
  `blk.0.ffn_up_shexp.weight`, and `blk.0.ffn_down_shexp.weight`, and compares
  against the oracle `ffn_shexp-0` payload. The 2048-value vector passes
  component numeric thresholds: max abs diff `1.4901161193847656e-08`, RMSE
  `3.087093312454213e-09`, cosine `0.9999999999999949`, mismatch count 0.
  This still does not run the full model loop or generate native candidate rows
- R1 engine-side MoE residual compare artifact
  `output/r1-engine-moe-residual-compare-20260627T093715Z/` stages the C++
  MoE residual compare to `ptl-cls-dvt2-008`, verifies normalized expert
  weighting, shared expert gating, `ffn_out-0`, and
  `attn_residual-0 + ffn_out-0 -> moe_residual-0`. The final 2048-value
  residual passes component numeric thresholds: max abs diff
  `2.682209014892578e-07`, RMSE `7.760108889930047e-09`, cosine
  `0.9999999999998894`, mismatch count 0. This still does not run the full
  model loop or generate native candidate rows
- R1 engine-side final norm compare artifact
  `output/r1-engine-final-norm-compare-20260627T094922Z/` stages the C++
  final RMSNorm compare to `ptl-cls-dvt2-008`, applies
  `output_norm.weight` to the global `l_out-39` payload, and compares against
  `result_norm`. The 2048-value vector passes component numeric thresholds:
  max abs diff `1.1444091796875e-05`, RMSE
  `8.575131329283268e-07`, cosine `0.9999999999999962`, mismatch count 0.
  This still does not run the full model loop or generate native candidate rows
- R1 engine-side LM head compare artifact
  `output/r1-engine-lm-head-compare-20260627T100103Z/` stages the C++
  LM head compare to `ptl-cls-dvt2-008`, applies `output.weight` to the global
  `result_norm` payload, and compares against `result_output` logits. The
  248320-value vector passes component numeric thresholds: max abs diff
  `2.86102294921875e-06`, RMSE `4.318970378914945e-07`, cosine
  `0.9999999999998903`, mismatch count 0. This still does not run the full
  model loop or generate native candidate rows
- R1 engine-side sampler compare artifact
  `output/r1-engine-sampler-compare-20260627T100947Z/` stages the C++
  deterministic top-k sampler compare to `ptl-cls-dvt2-008`, recomputes top-8
  rows from the oracle `result_output` logits, and compares against
  `sampler-topk.json`. Token ids match exactly with top token id `271`; max
  logit diff is `4.76837158203125e-05` under the JSON write precision
  threshold. This still does not run the full model loop or generate native
  candidate rows
- R1 engine-side linear attention delta compare artifact
  `output/r1-engine-linear-attn-delta-compare-20260627T112938Z/` stages the
  C++ L0 predelta gated-delta state update plus gated RMSNorm compare to
  `ptl-cls-dvt2-008`. It matches oracle `attn_output` with max abs diff
  `1.1920928955078125e-07` and `final_output` with max abs diff
  `5.960464477539062e-07`. This still does not implement the convolution
  state input path, full-attention KV updates, or generate native candidate rows
- R1 engine-side linear attention pre-conv compare artifact
  `output/r1-engine-linear-attn-preconv-compare-20260627T121529Z/` stages the
  C++ L0 pre-conv projection path from oracle `attn_norm`, computes
  `linear_attn_qkv_mixed`, alpha/beta, softplus/sigmoid gate values, and `z`.
  It matches `linear_attn_qkv_mixed` with max abs diff
  `3.814697265625e-06` and `z` with max abs diff
  `1.9073486328125e-06`. This still does not generate native candidate rows
- R1 engine-side linear attention conv-state compare artifact
  `output/r1-engine-linear-attn-conv-compare-20260627T123037Z/` stages the
  C++ L0 recurrent convolution state path to `ptl-cls-dvt2-008`, starts from
  the `short_math_001` prompt token ids, decodes embeddings, computes
  RMSNorm and pre-conv projections for tokens 0 through 15, maintains the
  `[oldest, middle, newest]` conv history, and compares token 15 against
  oracle payloads. `model_input_embed` matches exactly, `attention_norm` max
  abs diff is `1.71661376953125e-05`, `linear_attn_qkv_mixed` max abs diff is
  `2.288818359375e-05`, and `conv_output_raw` max abs diff is
  `1.811981201171875e-05`. This still does not generate native candidate rows
- R1 engine-side linear attention post-conv compare artifact
  `output/r1-engine-linear-attn-postconv-compare-20260627T114940Z/` stages
  the C++ L0 linear attention path from oracle `conv_output_raw`, computes
  alpha/beta/z, SILU, Q/K/V split, Q/K L2 norm, delta core, gated RMSNorm, and
  `ssm_out`. It matches `final_output` with max abs diff
  `5.960464477539062e-07` and `linear_attn_out` with max abs diff
  `1.043081283569336e-07`. This still does not implement the recurrent
  convolution state input path or generate native candidate rows
- R1 engine-side layer post-conv compare artifact
  `output/r1-engine-layer-postconv-compare-20260627T120307Z/` stitches L0
  native attention norm, linear attention post-conv, `ssm_out`,
  post-attention residual, and FFN/MoE from oracle `conv_output_raw` and
  `state_predelta`. Top-k matches exactly; `final_output` max abs diff is
  `4.6193599700927734e-07`, `linear_attention_out` max abs diff is
  `9.685754776000977e-08`, and `layer_output` max abs diff is
  `4.178308881819248e-05`. This still does not implement the recurrent
  convolution state input path or generate native candidate rows
- R1 engine-side stateful linear-attention layer compare artifact
  `output/r1-engine-layer-stateful-linear-attn-compare-20260627T124454Z/`
  replays the 16-token `short_math_001` prompt through native L0 embedding,
  RMSNorm, pre-conv projection, recurrent convolution state, gated-delta
  recurrent state, `ssm_out`, residual, and FFN/MoE. Token 15 matches oracle:
  `state_predelta` max abs diff is `1.6689300537109375e-05`,
  `final_output` max abs diff is `4.6193599700927734e-07`,
  `linear_attention_out` max abs diff is `1.1920928955078125e-07`, and
  `layer_output` max abs diff is `1.4901161193847656e-07`. This still does
  not implement full-attention KV updates or generate native candidate rows
- R1 engine-side L1 post-conv layer compare artifact
  `output/r1-engine-layer1-postconv-compare-20260627T130117Z/` reuses the
  parameterized layer core for layer 1 from oracle `l_out-0`,
  `conv_output_raw-1`, and `state_predelta-1`. Token 15 matches oracle:
  `final_output` max abs diff is `7.078051567077637e-08`,
  `linear_attention_out` max abs diff is `3.725290298461914e-08`, and
  `layer_output` max abs diff is `2.7008354663848877e-08`. Native/oracle
  router top-k exactly matches `[205,75,4,97,27,11,41,123]`. This is an
  accepted layer-core component, not an engine-maintained L1 state-update
  replay or native candidate row
- R1 engine-side L3 full-attention q/k/v projection compare artifact
  `output/r1-engine-full-attn-qkv-compare-20260627T131936Z/` validates the
  first full-attention layer boundary for `short_math_001` token 15 from
  oracle `l_out-2`. It matches `attn_norm-3`, `Qcur_full-3`,
  `Qcur_normed-3`, `Kcur-3`, `Kcur_normed-3`, and `Vcur-3`; max abs diffs are
  `7.62939453125e-06`, `3.814697265625e-06`,
  `2.86102294921875e-06`, `1.430511474609375e-06`,
  `2.1457672119140625e-06`, and `8.344650268554688e-07`. This covers the
  interleaved full-attention Q/gate split and per-head Q/K RMSNorm, not RoPE,
  KV cache update, attention output, or native candidate rows
- R1 engine-side L3 full-attention RoPE compare artifact
  `output/r1-engine-full-attn-rope-compare-20260627T134454Z/` validates the
  first full-attention layer IMRoPE boundary for `short_math_001` token 15
  from oracle `Qcur_normed-3` and `Kcur_normed-3`. It uses
  `rope_dimension_count=64`, sections `[11,11,10,0]`, position ids
  `[15,15,15,0]`, and `rope_freq_base=10000000`; `q_rope` max abs diff is
  `4.76837158203125e-07` and `k_rope` max abs diff is
  `2.384185791015625e-07`. This still does not update the full-attention KV
  cache, compute attention output, or emit native candidate rows
- R1 L3 full-attention history capture artifact
  `output/r1-full-attn-history-capture-20260627T142546Z/` captures
  `short_math_001` token positions 0 through 15 for layer 3, retaining only
  the Q/K/V and attention output-side tensors needed by the attention-core
  compare. Each token has `q_rope`, `k_rope`, `v`, `attn_pregate`,
  `attn_gated`, and `attn_output` payloads. This is oracle capture evidence,
  not native token correctness
- R1 engine-side L3 full-attention core compare artifact
  `output/r1-engine-full-attn-core-compare-20260627T143426Z/` validates
  causal attention for token 15 from captured `Qcur-3`, historical
  `Kcur-3`/`Vcur-3`, `head_dim=256`, `q_head_count=16`,
  `kv_head_count=2`, and scale `0.0625`. The selected f32 source-payload mode
  matches oracle `attn_pregate-3` with max abs diff
  `0.0017899274826049805`, RMSE `0.00018489847371300422`, cosine
  `0.99999986970357`, and mismatch count 0. This still does not implement
  native KV-cache updates or emit native candidate rows
- R1 engine-side L3 stateful full-attention layer compare artifact
  `output/r1-engine-full-attn-stateful-layer-compare-20260627T144639Z/`
  starts from oracle `l_out-2`, computes token15 Q/K/V/RoPE natively,
  appends native K/V to captured token0..14 history, and runs attention,
  gate, and output projection. The appended K max abs diff is
  `2.1457672119140625e-06`, appended V max abs diff is
  `8.344650268554688e-07`, pregate max abs diff is
  `0.0017900466918945312`, and final `attn_output-3` max abs diff is
  `0.00007873540744185448`. This validates a single-token K/V append path,
  but still consumes captured prior history and does not emit native
  candidate rows
- R1 all-layer full-attention history capture artifact
  `output/r1-full-attn-all-history-capture-20260627T145615Z/` captures
  `short_math_001` token positions 0 through 15 for full-attention layers
  `[3,7,11,15,19,23,27,31,35,39]`, retaining each layer's `q_rope`,
  `k_rope`, `v`, `attn_pregate`, `attn_gated`, and `attn_output`
  payloads. It contains 960 selected payloads and is oracle capture
  evidence, not native candidate JSONL evidence
- R1 engine-side all-layer stateful full-attention compare artifact
  `output/r1-engine-full-attn-all-stateful-layers-compare-20260627T151140Z/`
  validates token15 native K/V append plus causal attention, gate, and output
  projection for all 10 full-attention layers from captured token0..14 K/V
  history and oracle layer inputs. All 10 layers pass the component numeric
  gate with thresholds max abs `0.0125`, RMSE `0.001`, cosine `0.99998`;
  worst final `attn_output` max abs diff is `0.0028264522552490234`, RMSE
  is `0.00039453445936383765`, and cosine is `0.9999805655227852`. This
  still consumes captured prior history and oracle layer inputs, and does not
  emit native candidate rows
- R1 engine-side all-linear post-conv compare artifact
  `output/r1-engine-linear-attn-all-postconv-compare-20260627T153914Z/`
  validates token15 linear-attention post-conv math for all 30 linear layers
  using oracle `conv_output_raw` and `state_predelta` inputs. It stages 540
  payloads and passes strict component thresholds max abs `0.0005`, RMSE
  `0.00005`, cosine `0.99999`; worst `linear_attn_out` max abs diff is
  `0.00010106712579727173`. This does not prove convolution history replay,
  residual chaining, the integrated 40-layer loop, or native candidate rows
- R1 engine-side seed prompt input check artifact
  `output/r1-engine-seed-prompt-input-check-20260627T155328Z/` validates the
  native prompt-token input path for all six oracle seed rows. It stages
  `cases.tsv` plus six little-endian u32 token files, confirms 150 total
  prompt tokens, 98 unique token IDs/embedding rows, per-file FNV64 hashes,
  and exact `short_math_001` final-token embedding replay with max abs diff
  0, RMSE 0, cosine 1. This proves the engine can consume the fixed seed
  prompt token inputs; it does not emit native candidate rows
- R1 engine-side L3 full-attention gate compare artifact
  `output/r1-engine-full-attn-gate-compare-20260627T141107Z/` validates the
  per-head query/gate split from `Qcur_full-3` and applies
  `sigmoid(gate) * attn_pregate-3` to match oracle `attn_gated-3` for
  `short_math_001` token 15. It matches the 4096-value output with max abs
  diff `3.725290298461914e-09`, RMSE `2.558733662346639e-10`, cosine
  `0.9999999999999973`, and mismatch count 0. This uses oracle pregate
  attention state; it still does not update KV cache, run attention, or emit
  native candidate rows
- R1 engine-side L3 full-attention output projection compare artifact
  `output/r1-engine-full-attn-output-projection-compare-20260627T135544Z/`
  validates `blk.3.attn_output.weight` from oracle `attn_gated-3` to
  `attn_output-3` for `short_math_001` token 15. It matches the 2048-value
  output with max abs diff `2.2351741790771484e-08`, RMSE
  `2.0953007404094846e-09`, cosine `0.9999999999999576`, and mismatch count
  0. This validates the projection after attention using oracle attention
  state; it still does not update KV cache, run attention, or emit native
  candidate rows
- latest R1 native candidate route artifact
  `output/r1-native-candidate-route-20260627T155549Z/` registers 31
  component compare artifacts plus 1 prerequisite evidence artifact for the
  seed prompt input path. The component frontier covers the global sampler,
  L0 stateful linear-attention layer path, L1 post-conv layer core,
  all-30-linear token15 post-conv core evidence, the first full-attention
  q/k/v projection, RoPE, captured-history attention core, L3 stateful K/V
  append, gate, output projection boundaries, and all-10 full-attention
  stateful K/V append component evidence. It keeps
  `r1_native_correctness_gate_closed=false` and still requires a real
  `intel_qwen36_native` candidate JSONL for the six seed rows
- minimal O(1) engine skeleton and resident harness interface

R0 setup gates are closed for target/model facts, oracle bundle validation,
route feasibility, and the current resident harness load path. This is not an
optimized inference engine and no promoted speed claim exists in this repo.
The current selector gate is `r1_native_gguf_correctness_first_token_loop`.
