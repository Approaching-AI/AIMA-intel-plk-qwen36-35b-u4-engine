# GPU Backend Bring-Up

Workstream: `intel-qwen36-35b-a3b-gguf-q4km`

Status: active route. This is not a promoted speedup claim.

## Route

Backend route selected by user on 2026-06-29: GPU native backend on Arc B390.
CPU q4-plane micro-tuning is closed and preserved only as denominator/history.

First goal: reach or beat the same-host llama.cpp Vulkan decode floor,
`19.5 tok/s`, using the existing teacher-forced oracle and benchmark
discipline. `speedup_claims_allowed=false`.

## Initial Bring-Up Artifact

OpenCL device visibility on target:

- command: `ssh ptl-cls-dvt2-008 'command -v clinfo >/dev/null && clinfo -l || :'`
- result: `Intel(R) OpenCL Graphics` / `Intel(R) Arc(TM) B390 GPU`
- OpenCL loader observed: `/usr/lib/x86_64-linux-gnu/libOpenCL.so.1.0.0`

Sibling kernel corpus inventory:

- artifact: `output/gpu-opencl-corpus-inventory-20260629T161626Z/`
- source corpus:
  `/Users/jiawei-macmini/projects/intel-plk-highspeed/native/intel-plk-qwen36-native/src/gpu_opencl`
- files: `matvec_microbench.cpp`, `matvec_microbench.hpp`
- required checks passed
- OpenCL kernel entry points: `264`
- relevant mode strings: `638`
- q4 modes: `179`
- q6 modes: `215`
- MoE/down modes: `145`
- expert2pair / dualdot / weightfold modes: `9 / 7 / 5`

OpenCL runtime/source-stream gate:

- artifact: `output/gpu-opencl-runtime-source-stream-probe-20260629T162323Z/`
- tool: `tools/intel-qwen36-gpu-opencl-runtime-probe.py`
- source slice: first `67108864` bytes from the locked GGUF model
- required checks passed
- selected device: `Intel(R) Arc(TM) B390 GPU`
- kernel: `stream_checksum`
- checksum match: true (`7734207251`)
- event timing captured: true
- note: the `4.71163 GB/s` kernel figure is from a deliberately simple
  checksum kernel and is not a qmatvec/source-stream performance claim.

GPU raw-vs-repacked tensor stream gate:

- artifact: `output/gpu-repack-stream-probe-20260629T163140Z/`
- tool: `tools/intel-qwen36-gpu-repack-stream-probe.py`
- selected device: `Intel(R) Arc(TM) B390 GPU`
- selected tensors:
  - `blk.0.ffn_down_exps.weight`, `Q6_K`, `q6k_plane_v0`, raw/repacked
    `220.2 MB / 220.2 MB`
  - `blk.5.attn_qkv.weight`, `Q4_K`, `q4k_plane_v0`, raw/repacked
    `9.4 MB / 9.7 MB`
- required checks passed
- CPU/GPU checksum matched for raw, repacked, and quant-only streams
- aggregate raw/repacked bytes: `229638144 / 229900288`
- aggregate simple checksum kernel min GB/s mean: raw `5.5800`, repacked
  `5.6613`, quant-only `5.7822`
- note: this is a layout-stream plumbing gate. It does not prove qmatvec,
  decode, or model throughput.

GPU q4_K_8x8 packed stream gate:

- artifact: `output/gpu-repack-stream-probe-20260629T164218Z/`
- tool: `tools/intel-qwen36-gpu-repack-stream-probe.py`
- schema: `intel-qwen36-gpu-repack-stream-probe-v1`
- selected device: `Intel(R) Arc(TM) B390 GPU`
- required checks passed
- additional Q4 packed layout: `q4k_x8_llama_v0`, derived from local
  llama.cpp `block_q4_Kx8` / `repack<block_q4_K,8,8>` logic
- selected Q4 tensor: `blk.5.attn_qkv.weight`, dims `[2048,8192]`, raw bytes
  `9437184`, packed q4 x8 bytes `9437184`, overhead `1.0`
- CPU/GPU checksum matched for raw, plane-repacked, quant-only, and
  `q4k_x8_llama_v0` streams
- aggregate raw/plane bytes: `229638144 / 229900288`; q4 x8 packed bytes:
  `9437184`
- aggregate simple checksum kernel min GB/s mean: raw `5.4744`, plane
  `5.6479`, quant-only `5.6184`; q4 x8 stream `6.1434` for the selected Q4
  tensor
- note: this closes a packed-layout stream gate only. It does not prove GEMV,
  decode, or model throughput.

GPU q4_K_8x8 packed qmatvec profile gate:

- artifact: `output/gpu-q4x8-qmatvec-probe-20260629T171628Z/`
- tool: `tools/intel-qwen36-gpu-q4x8-qmatvec-probe.py`
- schema: `intel-qwen36-gpu-q4x8-qmatvec-probe-v2`
- engine shim: `engine/include/intel_qwen36/gpu_q4x8_matvec.hpp`,
  `engine/src/gpu_q4x8_matvec.cpp`
- OpenCL source: `engine/gpu/opencl/q4x8_matvec.cl`
- OpenCL source SHA256:
  `f71086c77765a6f3581702f45d696ef297ed3c68bcecf8bad010814a44607cc9`
- selected device: `Intel(R) Arc(TM) B390 GPU`
- selected tensor: `blk.5.attn_qkv.weight`, `Q4_K`, cols `2048`, rows
  `8192`, blocks/row `8`
- input quantization: deterministic synthetic `Q8_K` activation vector for a
  single-op gate
- required checks passed, including both GPU variants below
- q4 x8 packed bytes: `9437184`, equal to raw tensor bytes
- CPU packed vs CPU raw: relL2 `3.921738454e-06`, cosine `1`, max abs
  `1.049041748e-05`
- best GPU packed vs CPU packed: relL2 `2.843417873e-06`, cosine `1`, max abs
  `6.675720215e-06`
- profiled variants:
  - `group8_serial` (`1024` work-items, `8` rows/work-item): min
    `1393.333 us`, relL2 `2.675852548e-06`
  - `rowlane_parallel` (`8192` work-items, `1` row/work-item): min
    `161.979 us`, relL2 `2.843417873e-06`
- best variant: `rowlane_parallel`; kernel-only diagnostic packed bandwidth
  `58.26177467 GB/s`
- local engine build passed with the shim linked into `iq36_core`
- note: this closes a single-op correctness/timing/profile gate. It does not
  prove decode, token, or model throughput.

GPU Q4 attention/QKV shim consumer gate:

- artifact: `output/gpu-q4x8-qkv-consumer-probe-20260629T172456Z/`
- tool: `tools/intel-qwen36-gpu-q4x8-qkv-consumer-probe.py`
- schema: `intel-qwen36-gpu-q4x8-qkv-consumer-probe-v0`
- engine shim: `engine/include/intel_qwen36/gpu_q4x8_matvec.hpp`,
  `engine/src/gpu_q4x8_matvec.cpp`
- OpenCL source: `engine/gpu/opencl/q4x8_matvec.cl`
- oracle input/output:
  - `attn_norm-5__tok15__ord189.bin`
  - `linear_attn_qkv_mixed-5__tok15__ord190.bin`
- selected device: `Intel(R) Arc(TM) B390 GPU`
- selected tensor: `blk.5.attn_qkv.weight`, `Q4_K`, cols `2048`, rows
  `8192`, blocks/row `8`
- required checks passed
- CPU native vs oracle: max abs `1.907348633e-06`, RMSE
  `1.543678612e-07`, cosine `1`, mismatches `0`
- GPU rowlane vs CPU native: max abs `5.722045898e-06`, RMSE
  `3.112575961e-07`, cosine `1`, mismatches `0`
- GPU rowlane vs oracle: max abs `4.768371582e-06`, RMSE
  `2.961922258e-07`, cosine `1`, mismatches `0`
- rowlane kernel min: `170.833 us`; kernel-only diagnostic packed bandwidth:
  `55.24216047 GB/s`
- note: this closes the first Q4 attention/QKV shim consumer gate. It does not
  prove decode, token, or model throughput.

GPU Q4 x8 linear-attention preconv fan-in gate:

- artifact: `output/gpu-q4x8-preconv-fanin-probe-20260629T173320Z/`
- tool: `tools/intel-qwen36-gpu-q4x8-preconv-fanin-probe.py`
- schema: `intel-qwen36-gpu-q4x8-preconv-fanin-probe-v0`
- engine shim: `engine/include/intel_qwen36/gpu_q4x8_matvec.hpp`,
  `engine/src/gpu_q4x8_matvec.cpp`
- OpenCL source: `engine/gpu/opencl/q4x8_matvec.cl`
- selected device: `Intel(R) Arc(TM) B390 GPU`
- oracle payloads:
  - `attn_norm-5__tok15__ord189.bin`
  - `linear_attn_qkv_mixed-5__tok15__ord190.bin`
  - `alpha-5__tok15__ord198.bin`
  - `a_softplus-5__tok15__ord199.bin`
  - `gate-5__tok15__ord200.bin`
  - `beta-5__tok15__ord201.bin`
  - `beta_sigmoid-5__tok15__ord202.bin`
  - `z-5__tok15__ord205.bin`
- GPU projections from shared `attn_norm` input:
  - `blk.5.attn_qkv.weight`: rows `8192`, min `171.562 us`, GPU vs oracle
    max abs `4.768371582e-06`, RMSE `2.961922258e-07`
  - `blk.5.ssm_alpha.weight`: rows `32`, min `106.145 us`, GPU vs oracle
    max abs `8.344650269e-07`, RMSE `2.627353255e-07`
  - `blk.5.ssm_beta.weight`: rows `32`, min `107.083 us`, GPU vs oracle
    max abs `9.536743164e-07`, RMSE `2.813025881e-07`
  - `blk.5.attn_gate.weight`: rows `4096`, min `136.458 us`, GPU vs
    oracle max abs `2.861022949e-06`, RMSE `3.581015917e-07`
- derived CPU/GPU/oracle checks also passed for `a_softplus`, `gate`, and
  `beta_sigmoid`; all mismatches `0`, cosine `1`
- required checks passed
- note: this closes the multi-projection preconv fan-in component gate. It
  does not prove decode, token, or model throughput.

GPU Q4 x8 preconv-to-conv handoff gate:

- artifact: `output/gpu-q4x8-preconv-conv-handoff-probe-20260629T174804Z/`
- failed diagnostic predecessor:
  `output/gpu-q4x8-preconv-conv-handoff-probe-20260629T174418Z/`
- tool: `tools/intel-qwen36-gpu-q4x8-preconv-conv-handoff-probe.py`
- schema: `intel-qwen36-gpu-q4x8-preconv-conv-handoff-probe-v0`
- engine shim: `engine/include/intel_qwen36/gpu_q4x8_matvec.hpp`,
  `engine/src/gpu_q4x8_matvec.cpp`
- OpenCL source: `engine/gpu/opencl/q4x8_matvec.cl`
- added OpenCL kernel: `linear_attn_conv_f32`
- selected device: `Intel(R) Arc(TM) B390 GPU`
- required checks passed
- required comparison:
  - captured oracle `attn_norm-5__tok15__ord189.bin` feeds Q4 x8 QKV
  - GPU QKV vs oracle `linear_attn_qkv_mixed-5__tok15__ord190.bin`: max
    abs `4.768371582e-06`, RMSE `2.961922258e-07`
  - a valid layer-5 conv state is reconstructed by CPU native prompt replay
  - GPU conv output vs CPU component: max abs `9.536743164e-07`, RMSE
    `3.50922206e-08`
  - GPU next conv state vs CPU component: max abs `5.722045898e-06`
- timings: QKV rowlane min `224.375 us`; F32 depthwise conv min `6.666 us`
- diagnostic only:
  - captured `conv_output_raw-5__tok15__ord191.bin` is not a required oracle
    check because the capture bundle does not include pre-token conv history
    state
  - CPU replay frontier vs captured `attn_norm-5` failed diagnostic:
    max abs `0.1403956413`, RMSE `0.03621432816`
  - GPU conv output vs captured `conv_output_raw-5` diagnostic max abs
    `0.04150788486`, RMSE `0.002165880509`
- note: this closes the device handoff component gate for oracle QKV input
  into GPU conv. It does not prove decode, token, or model throughput.

GPU Q4 x8 postconv prep handoff gate:

- artifact: `output/gpu-q4x8-postconv-prep-probe-20260629T175929Z/`
- tool: `tools/intel-qwen36-gpu-q4x8-postconv-prep-probe.py`
- schema: `intel-qwen36-gpu-q4x8-postconv-prep-probe-v0`
- engine shim: `engine/include/intel_qwen36/gpu_q4x8_matvec.hpp`,
  `engine/src/gpu_q4x8_matvec.cpp`
- OpenCL source: `engine/gpu/opencl/q4x8_matvec.cl`
- OpenCL source SHA256:
  `0f7eda8fa1c512f29817fef66e39d77b997d8463cff0161e829d389669a5f53a`
- added OpenCL kernels: `linear_attn_postconv_silu_split_f32`,
  `linear_attn_l2_norm_heads_f32`
- selected device: `Intel(R) Arc(TM) B390 GPU`
- input oracle payload: `conv_output_raw-5__tok15__ord191.bin`
- required checks passed
- GPU vs oracle max abs/RMSE:
  - `conv_output_silu`: `2.384185791e-07` / `5.045548752e-09`
  - `q_conv`: `1.192092896e-07` / `4.616882076e-09`
  - `q_conv_predelta`: `1.788139343e-07` / `8.06136038e-09`
  - `k_conv`: `5.960464478e-08` / `5.587645889e-09`
  - `k_conv_predelta`: `1.192092896e-07` / `1.042985742e-08`
  - `v_conv_predelta`: `2.384185791e-07` / `4.964517204e-09`
- timings: SiLU/split min `3.333 us`, Q L2 min `44.166 us`, K L2 min
  `44.895 us`
- note: this closes the postconv prep component gate from captured
  `conv_output_raw` through SiLU, Q/K/V split, and Q/K L2 normalization. It
  does not prove decode, token, or model throughput.

GPU Q4 x8 delta recurrent handoff gate:

- artifact: `output/gpu-q4x8-delta-recurrent-probe-20260629T181034Z/`
- tool: `tools/intel-qwen36-gpu-q4x8-delta-recurrent-probe.py`
- schema: `intel-qwen36-gpu-q4x8-delta-recurrent-probe-v0`
- engine shim: `engine/include/intel_qwen36/gpu_q4x8_matvec.hpp`,
  `engine/src/gpu_q4x8_matvec.cpp`
- OpenCL source: `engine/gpu/opencl/q4x8_matvec.cl`
- OpenCL source SHA256:
  `82992757355e446ebb88ff44785ebfe95fe8f0e4b0d279f215971d9f5cfcc1b6`
- added OpenCL kernels: `linear_attn_delta_recurrent_f32`,
  `linear_attn_final_norm_f32`
- selected device: `Intel(R) Arc(TM) B390 GPU`
- input oracle payloads: `q_conv_predelta-5__tok15__ord194.bin`,
  `k_conv_predelta-5__tok15__ord196.bin`,
  `v_conv_predelta-5__tok15__ord197.bin`, `gate-5__tok15__ord200.bin`,
  `beta_sigmoid-5__tok15__ord202.bin`,
  `state_predelta-5__tok15__ord203.bin`, and `z-5__tok15__ord205.bin`
- required checks passed
- required comparison max abs/RMSE:
  - GPU `attention_output` vs oracle `attn_output-5__tok15__ord204.bin`:
    `1.11758709e-08` / `2.556845485e-10`
  - GPU `final_output` vs oracle `final_output-5__tok15__ord206.bin`:
    `2.235174179e-08` / `1.851338136e-09`
  - GPU next recurrent state vs CPU component: `2.384185791e-07` /
    `9.805425517e-10`
- timings: recurrent delta min `271.354 us`, final gated RMS norm min
  `114.27 us`
- note: this closes the recurrent update, attention output, and final gated RMS
  output component gate. It does not prove decode, token, or model throughput.

GPU Q4 x8 output projection handoff gate:

- artifact: `output/gpu-q4x8-output-projection-probe-20260629T181533Z/`
- tool: `tools/intel-qwen36-gpu-q4x8-output-projection-probe.py`
- schema: `intel-qwen36-gpu-q4x8-output-projection-probe-v0`
- engine shim: `engine/include/intel_qwen36/gpu_q4x8_matvec.hpp`,
  `engine/src/gpu_q4x8_matvec.cpp`
- OpenCL source: `engine/gpu/opencl/q4x8_matvec.cl`
- OpenCL source SHA256:
  `82992757355e446ebb88ff44785ebfe95fe8f0e4b0d279f215971d9f5cfcc1b6`
- selected device: `Intel(R) Arc(TM) B390 GPU`
- input oracle payload: `final_output-5__tok15__ord206.bin`
- output oracle payload: `linear_attn_out-5__tok15__ord207.bin`
- selected tensor: `blk.5.ssm_out.weight`, `Q4_K`, cols `4096`, rows `2048`
- required checks passed
- GPU `linear_attn_out` vs oracle max abs/RMSE:
  `4.470348358e-08` / `5.043069309e-09`
- GPU `linear_attn_out` vs CPU native max abs/RMSE:
  `4.097819328e-08` / `5.184632851e-09`
- timing: output projection rowlane min `507.5 us`; kernel-only diagnostic
  packed bandwidth `9.297718227 GB/s`
- note: this closes the Q4 x8 `ssm_out.weight` projection from captured
  `final_output` to captured `linear_attn_out`. It does not prove decode,
  token, or model throughput.

GPU F32 FFN/MoE router handoff gate:

- artifact: `output/gpu-f32-router-probe-20260629T182846Z/`
- tool: `tools/intel-qwen36-gpu-f32-router-probe.py`
- schema: `intel-qwen36-gpu-f32-router-probe-v0`
- engine shim: `engine/include/intel_qwen36/gpu_q4x8_matvec.hpp`,
  `engine/src/gpu_q4x8_matvec.cpp`
- OpenCL source: `engine/gpu/opencl/q4x8_matvec.cl`
- OpenCL source SHA256:
  `f6cf562cef5a6045fd91008030243efdbbe50d7decbe05a356565739ef50d1f1`
- added OpenCL kernel: `f32_matvec_row_f32`
- selected device: `Intel(R) Arc(TM) B390 GPU`
- input oracle payload: `attn_post_norm-5__tok15__ord209.bin`
- selected tensor: `blk.5.ffn_gate_inp.weight`, `F32`, cols `2048`, rows
  `256`
- required checks passed
- GPU router logits vs oracle max abs/RMSE:
  `1.287460327e-05` / `3.846549395e-06`
- GPU-derived top-k vs oracle mismatches: `0`
- GPU normalized router weights vs oracle max abs/RMSE:
  `3.576278687e-07` / `1.764555232e-07`
- timing: router logits F32 matvec min `371.875 us`; kernel-only diagnostic
  weight bandwidth `5.639400336 GB/s`
- note: this closes router logits and CPU/GPU-derived softmax/top-k/weights
  against captured `ffn_moe_*` oracle payloads. It does not prove decode,
  token, or model throughput.

GPU Q4 x8 selected expert gate-up handoff gate:

- artifact: `output/gpu-q4x8-selected-gate-up-probe-20260629T183548Z/`
- tool: `tools/intel-qwen36-gpu-q4x8-selected-gate-up-probe.py`
- schema: `intel-qwen36-gpu-q4x8-selected-gate-up-probe-v0`
- engine shim: `engine/include/intel_qwen36/gpu_q4x8_matvec.hpp`,
  `engine/src/gpu_q4x8_matvec.cpp`
- OpenCL source: `engine/gpu/opencl/q4x8_matvec.cl`
- OpenCL source SHA256:
  `f6cf562cef5a6045fd91008030243efdbbe50d7decbe05a356565739ef50d1f1`
- selected device: `Intel(R) Arc(TM) B390 GPU`
- input oracle payloads: `attn_post_norm-5__tok15__ord209.bin`,
  `ffn_moe_topk-5__tok15__ord212.bin`
- output oracle payload: `ffn_moe_gate_up-5__tok15__ord215.bin`
- selected tensor: `blk.5.ffn_gate_up_exps.weight`, `Q4_K`, cols `2048`,
  rows per expert `1024`, selected rows `8192`
- selected expert ids: `[151, 88, 104, 251, 46, 103, 186, 218]`
- required checks passed
- GPU selected gate-up vs oracle max abs/RMSE:
  `8.344650269e-07` / `1.500601006e-07`
- GPU selected gate-up vs CPU native max abs/RMSE:
  `9.536743164e-07` / `1.520374886e-07`
- timing: selected gate-up rowlane min `264.687 us`; kernel-only diagnostic
  packed bandwidth `35.65412733 GB/s`
- note: this closes selected top-k expert slice packing plus Q4 x8 matvec for
  `ffn_gate_up_exps.weight`. It does not prove decode, token, or model
  throughput.

GPU selected expert SwiGLU handoff gate:

- artifact: `output/gpu-selected-swiglu-probe-20260629T184456Z/`
- tool: `tools/intel-qwen36-gpu-selected-swiglu-probe.py`
- schema: `intel-qwen36-gpu-selected-swiglu-probe-v0`
- OpenCL source: `engine/gpu/opencl/q4x8_matvec.cl`
- OpenCL source SHA256:
  `e07a441ffa3c9d784839d451a42e60985fb087a75667982fa99224d51fd71f57`
- added OpenCL kernel: `ffn_moe_swiglu_f32`
- selected device: `Intel(R) Arc(TM) B390 GPU`
- input oracle payloads: `ffn_moe_topk-5__tok15__ord212.bin`,
  `ffn_moe_weights_norm-5__tok15__ord214.bin`,
  `ffn_moe_gate_up-5__tok15__ord215.bin`
- output oracle payload: `ffn_moe_swiglu-5__tok15__ord218.bin`
- selected tensor: `blk.5.ffn_gate_up_exps.weight`, `Q4_K`, intermediate
  `512`, selected experts `8`
- selected expert ids: `[151, 88, 104, 251, 46, 103, 186, 218]`
- required checks passed
- GPU selected SwiGLU vs oracle max abs/RMSE:
  `5.960464478e-08` / `4.40002369e-09`
- GPU selected SwiGLU vs CPU native max abs/RMSE:
  `2.980232239e-08` / `3.771375142e-09`
- timing: selected SwiGLU kernel min `3.437 us`; kernel-only diagnostic IO
  bandwidth `14.30084376 GB/s`
- note: this closes the captured selected gate/up activation to selected
  SwiGLU component. The probe uses a narrow probe-local OpenCL runner to avoid
  growing the C++ shim for an elementwise handoff. It does not prove decode,
  token, or model throughput.

GPU selected expert down handoff gate:

- artifact: `output/gpu-selected-down-probe-20260629T190155Z/`
- tool: `tools/intel-qwen36-gpu-selected-down-probe.py`
- schema: `intel-qwen36-gpu-selected-down-probe-v0`
- engine shim reused for Q4 path: `engine/include/intel_qwen36/gpu_q4x8_matvec.hpp`,
  `engine/src/gpu_q4x8_matvec.cpp`
- OpenCL source: `engine/gpu/opencl/q4x8_matvec.cl`
- OpenCL source SHA256:
  `ab12b4825d69c652f8a253a5b84397a382973b0e2001460195d370133462cc9d`
- added OpenCL kernel: `q6k_selected_down_matvec_row` (compiled here, reserved
  for Q6 selected-down tensors)
- selected device: `Intel(R) Arc(TM) B390 GPU`
- input oracle payloads: `ffn_moe_swiglu-5__tok15__ord218.bin`,
  `ffn_moe_topk-5__tok15__ord212.bin`
- output oracle payload: `ffn_moe_down-5__tok15__ord219.bin`
- selected tensor: `blk.5.ffn_down_exps.weight`, `Q4_K`, cols `512`, rows per
  expert `2048`, selected rows `16384`
- selected expert ids: `[151, 88, 104, 251, 46, 103, 186, 218]`
- required checks passed
- GPU selected down vs oracle max abs/RMSE:
  `4.470348358e-08` / `4.109370833e-09`
- GPU selected down vs CPU native max abs/RMSE:
  `4.470348358e-08` / `4.562161527e-09`
- timing: selected down rowlane min `591.663 us` across `8` expert launches;
  kernel-only diagnostic raw bandwidth `7.975134494 GB/s`
- note: this closes selected expert down for the captured layer-5 `Q4_K`
  tensor using one Q8 activation per selected expert. It does not prove decode,
  token, or model throughput.

GPU selected MoE weighted aggregation handoff gate:

- artifact: `output/gpu-moe-weighted-aggregate-probe-20260629T191043Z/`
- tool: `tools/intel-qwen36-gpu-moe-weighted-aggregate-probe.py`
- schema: `intel-qwen36-gpu-moe-weighted-aggregate-probe-v0`
- OpenCL source: `engine/gpu/opencl/q4x8_matvec.cl`
- OpenCL source SHA256:
  `b1fa55873c90747d83c3c902e23d67a535142cd8e35a4b7917193f520d8708c3`
- added OpenCL kernel: `ffn_moe_weighted_aggregate_f32`
- selected device: `Intel(R) Arc(TM) B390 GPU`
- input oracle payloads: `ffn_moe_down-5__tok15__ord219.bin`,
  `ffn_moe_weights_norm-5__tok15__ord214.bin`
- output oracle payloads: `ffn_moe_weighted-5__tok15__ord220.bin`,
  `ffn_moe_out-5__tok15__ord221.bin`
- shape: hidden `2048`, selected experts `8`, weighted values `16384`
- required checks passed
- GPU weighted vs oracle max abs/RMSE: `0` / `0`
- GPU MoE out vs oracle max abs/RMSE: `0` / `0`
- GPU weighted and MoE out vs CPU native max abs/RMSE: `0` / `0`
- timing: weighted aggregate kernel min `8.541 us`; kernel-only diagnostic IO
  bandwidth `16.3090973 GB/s`
- note: this closes router-weight scaling plus selected expert aggregation for
  the captured layer-5 selected MoE branch. It does not prove decode, token, or
  model throughput.

GPU shared expert gate handoff gate:

- artifact: `output/gpu-shared-expert-gate-probe-20260629T192104Z/`
- tool: `tools/intel-qwen36-gpu-shared-expert-gate-probe.py`
- schema: `intel-qwen36-gpu-shared-expert-gate-probe-v0`
- OpenCL source: `engine/gpu/opencl/q4x8_matvec.cl`
- OpenCL source SHA256:
  `3f4e180683c4ae40ff3c5f571f60c01b657a1ea4675be5fd248d0f8410b73be6`
- added OpenCL kernel: `shared_expert_gate_apply_f32`
- reused OpenCL kernel: `f32_matvec_row_f32`
- selected device: `Intel(R) Arc(TM) B390 GPU`
- input oracle payloads: `attn_post_norm-5__tok15__ord209.bin`,
  `ffn_shexp-5__tok15__ord222.bin`
- output oracle payloads: `shared_expert_gate-5__tok15__ord223.bin`,
  `shared_expert_gate_sigmoid-5__tok15__ord224.bin`,
  `ffn_shexp_gated-5__tok15__ord225.bin`
- selected tensor: `blk.5.ffn_gate_inp_shexp.weight`, `F32`, values `2048`
- required checks passed
- GPU shared gate vs oracle max abs/RMSE:
  `1.192092896e-07` / `1.192092896e-07`
- GPU shared gate sigmoid vs oracle max abs/RMSE: `0` / `0`
- GPU shared gated output vs oracle max abs/RMSE: `0` / `0`
- timing: shared gate F32 matvec min `284.583 us`; shared gate apply min
  `3.02 us`
- note: this closes the scalar shared-expert input gate, sigmoid, and captured
  shared-expert output multiply for layer 5. It does not prove decode, token,
  or model throughput.

GPU final FFN output add handoff gate:

- artifact: `output/gpu-ffn-output-add-probe-20260629T192821Z/`
- tool: `tools/intel-qwen36-gpu-ffn-output-add-probe.py`
- schema: `intel-qwen36-gpu-ffn-output-add-probe-v0`
- OpenCL source: `engine/gpu/opencl/q4x8_matvec.cl`
- OpenCL source SHA256:
  `5f6b7557f34c3aa2c66e82b854d6078dcb9bc26c24df2c140923802632018186`
- added OpenCL kernel: `ffn_output_add_f32`
- selected device: `Intel(R) Arc(TM) B390 GPU`
- input oracle payloads: `ffn_moe_out-5__tok15__ord221.bin`,
  `ffn_shexp_gated-5__tok15__ord225.bin`
- output oracle payload: `ffn_out-5__tok15__ord226.bin`
- shape: hidden `2048`
- required checks passed
- GPU final FFN output vs oracle max abs/RMSE: `0` / `0`
- GPU final FFN output vs CPU native max abs/RMSE: `0` / `0`
- timing: final FFN output add kernel min `2.291 us`; kernel-only diagnostic IO
  bandwidth `10.72719337 GB/s`
- note: this closes selected MoE output plus gated shared expert output into
  captured `ffn_out` for layer 5. It does not prove decode, token, or model
  throughput.

GPU post-FFN residual add handoff gate:

- artifact: `output/gpu-post-ffn-residual-add-probe-20260629T193307Z/`
- tool: `tools/intel-qwen36-gpu-post-ffn-residual-add-probe.py`
- schema: `intel-qwen36-gpu-post-ffn-residual-add-probe-v0`
- OpenCL source: `engine/gpu/opencl/q4x8_matvec.cl`
- OpenCL source SHA256:
  `66d8726ca3e80e5decb27c29b186943471700ebe88b8644e903f6932bb189fce`
- added OpenCL kernel: `post_ffn_residual_add_f32`
- selected device: `Intel(R) Arc(TM) B390 GPU`
- input oracle payloads: `attn_residual-5__tok15__ord208.bin`,
  `ffn_out-5__tok15__ord226.bin`
- output oracle payload: `l_out-5__tok15__ord227.bin`
- shape: hidden `2048`
- required checks passed
- GPU layer output vs oracle max abs/RMSE: `0` / `0`
- GPU layer output vs CPU native max abs/RMSE: `0` / `0`
- timing: post-FFN residual add kernel min `2.291 us`; kernel-only diagnostic
  IO bandwidth `10.72719337 GB/s`
- note: this closes final FFN output plus attention residual into captured
  `l_out` for layer 5. It does not prove decode, token, or model throughput.

GPU captured layer shell handoff gate:

- artifact: `output/gpu-captured-layer-shell-probe-20260629T194314Z/`
- tool: `tools/intel-qwen36-gpu-captured-layer-shell-probe.py`
- schema: `intel-qwen36-gpu-captured-layer-shell-probe-v0`
- OpenCL source: `engine/gpu/opencl/q4x8_matvec.cl`
- OpenCL source SHA256:
  `66d8726ca3e80e5decb27c29b186943471700ebe88b8644e903f6932bb189fce`
- reused OpenCL kernels: `ffn_moe_weighted_aggregate_f32`,
  `f32_matvec_row_f32`, `shared_expert_gate_apply_f32`,
  `ffn_output_add_f32`, `post_ffn_residual_add_f32`
- selected device: `Intel(R) Arc(TM) B390 GPU`
- input oracle payloads: `attn_residual-5__tok15__ord208.bin`,
  `attn_post_norm-5__tok15__ord209.bin`,
  `ffn_moe_weights_norm-5__tok15__ord214.bin`,
  `ffn_moe_down-5__tok15__ord219.bin`,
  `ffn_shexp-5__tok15__ord222.bin`
- output oracle payloads: `ffn_moe_weighted-5__tok15__ord220.bin`,
  `ffn_moe_out-5__tok15__ord221.bin`,
  `shared_expert_gate-5__tok15__ord223.bin`,
  `shared_expert_gate_sigmoid-5__tok15__ord224.bin`,
  `ffn_shexp_gated-5__tok15__ord225.bin`,
  `ffn_out-5__tok15__ord226.bin`, `l_out-5__tok15__ord227.bin`
- shape: hidden `2048`, selected experts `8`, weighted values `16384`
- required checks passed
- GPU weighted, MoE output, final FFN output, and layer output vs oracle max
  abs/RMSE: `0` / `0`
- GPU shared gate vs oracle max abs/RMSE:
  `1.192092896e-07` / `1.192092896e-07`
- GPU shared gate sigmoid and shared gated output vs oracle max abs/RMSE:
  `0` / `0`
- timing: captured layer shell kernel sum min `271.352 us`; mean
  `278.0563636 us`
- note: this composes the already closed layer-5 post-attention/FFN component
  kernels from captured boundaries into one target-side shell that emits
  captured `l_out`. It does not recompute attention, selected expert matvecs,
  prompt state, or token decode.

GPU resident captured-layer shell handoff gate:

- artifact:
  `output/gpu-resident-captured-layer-shell-probe-20260629T195135Z/`
- tool: `tools/intel-qwen36-gpu-resident-captured-layer-shell-probe.py`
- schema: `intel-qwen36-gpu-resident-captured-layer-shell-probe-v0`
- OpenCL source: `engine/gpu/opencl/q4x8_matvec.cl`
- OpenCL source SHA256:
  `66d8726ca3e80e5decb27c29b186943471700ebe88b8644e903f6932bb189fce`
- resident API: `captured_layer_shell_load_once_run_many`
- resident load count: `1`
- resident shell invocations: `11`
- resident reuse evidence: OpenCL program `true`, device buffers `true`
- selected device: `Intel(R) Arc(TM) B390 GPU`
- shape: hidden `2048`, selected experts `8`, weighted values `16384`
- required checks passed
- GPU weighted, MoE output, final FFN output, and layer output vs oracle max
  abs/RMSE: `0` / `0`
- GPU shared gate vs oracle max abs/RMSE:
  `1.192092896e-07` / `1.192092896e-07`
- GPU shared gate sigmoid and shared gated output vs oracle max abs/RMSE:
  `0` / `0`
- timing: resident captured layer shell kernel sum min `248.228 us`; mean
  `252.7729091 us`
- note: this keeps the captured single-layer shell in one target-side process:
  model metadata, OpenCL program, payload buffers, and scratch stay resident
  across repeated shell invocations. It remains captured boundary evidence,
  not prompt/token decode or model throughput.

GPU resident selected-expert FFN shell handoff gate:

- artifact:
  `output/gpu-resident-selected-ffn-shell-probe-20260629T200646Z/`
- tool: `tools/intel-qwen36-gpu-resident-selected-ffn-shell-probe.py`
- schema: `intel-qwen36-gpu-resident-selected-ffn-shell-probe-v0`
- OpenCL source: `engine/gpu/opencl/q4x8_matvec.cl`
- OpenCL source SHA256:
  `66d8726ca3e80e5decb27c29b186943471700ebe88b8644e903f6932bb189fce`
- resident API: `selected_expert_ffn_load_once_run_many`
- resident load count: `1`
- resident shell invocations: `5`
- selected tensors: `blk.5.ffn_gate_up_exps.weight` `Q4_K`,
  `blk.5.ffn_down_exps.weight` `Q4_K`
- selected down Q4 expert launches: `8`
- selected down host Q8 bridge: `true`
- selected device: `Intel(R) Arc(TM) B390 GPU`
- required checks passed
- GPU selected gate-up vs oracle max abs/RMSE:
  `8.344650269e-07` / `1.500601006e-07`
- GPU selected SwiGLU vs oracle max abs/RMSE:
  `3.576278687e-07` / `2.934449173e-08`
- GPU selected down vs oracle max abs/RMSE:
  `4.470348358e-08` / `6.161441542e-09`
- GPU MoE output vs oracle max abs/RMSE:
  `1.490116119e-08` / `4.093183861e-09`
- GPU layer output vs oracle max abs/RMSE:
  `1.490116119e-08` / `4.22913276e-09`
- timing: selected FFN kernel sum min `803.225 us`; resident
  selected-FFN-to-layer kernel sum min `1074.577 us`
- note: this expands the resident layer-5 shell upstream from captured
  `ffn_moe_down` through selected gate-up, selected SwiGLU, selected down,
  weighted aggregation, shared gate/apply, FFN add, and residual. The shared
  expert output itself remains captured, and selected-down Q4 still uses a
  host Q8 activation bridge per selected expert. It does not prove prompt/token
  decode or model throughput.

GPU resident shared-expert FFN shell handoff gate:

- artifact:
  `output/gpu-resident-shared-ffn-shell-probe-20260629T201810Z/`
- tool: `tools/intel-qwen36-gpu-resident-shared-ffn-shell-probe.py`
- schema: `intel-qwen36-gpu-resident-shared-ffn-shell-probe-v0`
- OpenCL source: `engine/gpu/opencl/q4x8_matvec.cl`
- OpenCL source SHA256:
  `66d8726ca3e80e5decb27c29b186943471700ebe88b8644e903f6932bb189fce`
- resident API: `shared_expert_ffn_load_once_run_many`
- resident load count: `1`
- resident shell invocations: `5`
- selected tensors: `blk.5.ffn_gate_up_exps.weight` `Q4_K`,
  `blk.5.ffn_down_exps.weight` `Q4_K`
- shared tensors: `blk.5.ffn_gate_shexp.weight` `Q4_K`,
  `blk.5.ffn_up_shexp.weight` `Q4_K`,
  `blk.5.ffn_down_shexp.weight` `Q4_K`
- selected down Q4 expert launches: `8`
- selected down host Q8 bridge: `true`
- shared down host Q8 bridge: `true`
- selected device: `Intel(R) Arc(TM) B390 GPU`
- required checks passed
- GPU selected down vs oracle max abs/RMSE:
  `4.470348358e-08` / `6.161441542e-09`
- GPU shared down vs oracle max abs/RMSE:
  `4.470348358e-08` / `4.849848113e-09`
- GPU MoE output vs oracle max abs/RMSE:
  `1.490116119e-08` / `4.093183861e-09`
- GPU shared gated output vs oracle max abs/RMSE:
  `5.587935448e-09` / `7.128219373e-10`
- GPU final FFN output vs oracle max abs/RMSE:
  `1.490116119e-08` / `4.173702909e-09`
- GPU layer output vs oracle max abs/RMSE:
  `1.490116119e-08` / `4.309118903e-09`
- timing: selected FFN kernel sum min `736.661 us`; shared FFN kernel
  sum min `218.54 us`; resident full-FFN-to-layer kernel sum min
  `1082.49 us`
- note: this expands the resident layer-5 shell upstream from captured
  `ffn_shexp` through shared gate/up, shared SwiGLU, shared down, shared input
  gate/apply, FFN add, and residual. Both selected-down and shared-down Q4
  paths still use explicit host Q8 activation bridges. It does not prove
  prompt/token decode or model throughput.

GPU resident attention-to-FFN layer shell handoff gate:

- artifact:
  `output/gpu-resident-attention-ffn-shell-probe-20260629T203045Z/`
- tool: `tools/intel-qwen36-gpu-resident-attention-ffn-shell-probe.py`
- schema: `intel-qwen36-gpu-resident-attention-ffn-shell-probe-v0`
- OpenCL source: `engine/gpu/opencl/q4x8_matvec.cl`
- OpenCL source SHA256:
  `66d8726ca3e80e5decb27c29b186943471700ebe88b8644e903f6932bb189fce`
- embedded probe OpenCL SHA256:
  `cb6b518694e13b95887ca79bf17019bc3372222cfbd0ab1bfbb6e6ce708ef89a`
- probe-extra OpenCL kernel: `rms_norm_hidden_f32`
- resident API: `attention_to_ffn_layer_shell_load_once_run_many`
- resident load count: `1`
- resident shell invocations: `5`
- selected device: `Intel(R) Arc(TM) B390 GPU`
- required checks passed
- attention output projection host Q8 bridge: `true`
- attention front host boundary between Q4 and F32 kernels: `true`
- selected down host Q8 bridge: `true`
- shared down host Q8 bridge: `true`
- GPU linear attention output vs oracle max abs/RMSE:
  `4.470348358e-08` / `5.043069309e-09`
- GPU post-attention residual vs oracle max abs/RMSE:
  `5.960464478e-08` / `5.137625364e-09`
- GPU FFN RMSNorm (`attn_post_norm`) vs oracle max abs/RMSE:
  `1.907348633e-06` / `2.634980602e-07`
- GPU selected down vs oracle max abs/RMSE:
  `8.940696716e-08` / `1.251804678e-08`
- GPU shared down vs oracle max abs/RMSE:
  `1.043081284e-07` / `1.511338098e-08`
- GPU final FFN output vs oracle max abs/RMSE:
  `4.470348358e-08` / `9.104029818e-09`
- GPU layer output vs oracle max abs/RMSE:
  `5.960464478e-08` / `1.043267898e-08`
- timing: attention front kernel sum min `550.936 us`; selected FFN kernel
  sum min `468.956 us`; shared FFN kernel sum min `274.269 us`; FFN tail
  kernel sum min `149.685 us`; resident attention-to-layer kernel sum min
  `1443.846 us`
- note: this expands the resident layer-5 shell upstream from captured
  `attn_post_norm`/`attn_residual` to captured linear-attention `final_output`
  and previous layer residual input. It computes `ssm_out.weight`,
  post-attention residual, FFN RMSNorm, selected/shared FFN branches, tail
  aggregation, and residual to captured `l_out`. The Q4 output projection and
  selected/shared down Q4 paths still use explicit host Q8 activation bridges,
  and the attention front currently has an explicit host boundary between the
  Q4 matvec runner and F32 residual/RMSNorm kernels. It does not prove
  prompt/token decode or model throughput.

GPU resident postconv-to-layer shell handoff gate:

- artifact:
  `output/gpu-resident-postconv-layer-shell-probe-20260629T204117Z/`
- tool: `tools/intel-qwen36-gpu-resident-postconv-layer-shell-probe.py`
- schema: `intel-qwen36-gpu-resident-postconv-layer-shell-probe-v0`
- OpenCL source: `engine/gpu/opencl/q4x8_matvec.cl`
- OpenCL source SHA256:
  `66d8726ca3e80e5decb27c29b186943471700ebe88b8644e903f6932bb189fce`
- embedded probe OpenCL SHA256:
  `cb6b518694e13b95887ca79bf17019bc3372222cfbd0ab1bfbb6e6ce708ef89a`
- probe-extra OpenCL kernel: `rms_norm_hidden_f32`
- resident API: `postconv_to_layer_shell_load_once_run_many`
- resident load count: `1`
- resident shell invocations: `5`
- selected device: `Intel(R) Arc(TM) B390 GPU`
- required checks passed
- delta-to-attention host boundary: `true`
- attention output projection host Q8 bridge: `true`
- attention front host boundary between Q4 and F32 kernels: `true`
- selected down host Q8 bridge: `true`
- shared down host Q8 bridge: `true`
- GPU attention output vs oracle max abs/RMSE:
  `1.11758709e-08` / `2.556845485e-10`
- GPU linear-attention final output vs oracle max abs/RMSE:
  `2.235174179e-08` / `1.851338136e-09`
- GPU post-attention residual vs oracle max abs/RMSE:
  `5.960464478e-08` / `5.554601572e-09`
- GPU FFN RMSNorm (`attn_post_norm`) vs oracle max abs/RMSE:
  `1.907348633e-06` / `2.738521503e-07`
- GPU final FFN output vs oracle max abs/RMSE:
  `6.705522537e-08` / `8.859113151e-09`
- GPU layer output vs oracle max abs/RMSE:
  `5.960464478e-08` / `1.040317578e-08`
- timing: delta-to-final kernel sum min `173.957 us`; attention front
  kernel sum min `522.29 us`; output projection min `283.02 us`; FFN RMSNorm
  min `237.812 us`; selected FFN kernel sum min `455.412 us`; shared FFN
  kernel sum min `262.603 us`; FFN tail kernel sum min `142.81 us`;
  resident postconv-to-layer kernel sum min `1557.072 us`
- note: this expands the resident layer-5 shell upstream from captured
  postconv predelta outputs, gates, `z`, and recurrent state through delta
  recurrent, final norm, output projection, post-attention residual, FFN
  RMSNorm, selected/shared FFN branches, tail aggregation, and residual to
  captured `l_out`. The delta-to-attention host boundary and all Q8 activation
  bridges remain explicit. This is captured single-layer evidence only; it does
  not prove prompt/token decode or model throughput.

GPU conv-history state capture handoff gate:

- artifact: `output/gpu-conv-history-state-capture-probe-20260629T205556Z/`
- nested capture artifact:
  `output/gpu-conv-history-state-capture-probe-20260629T205556Z/capture/`
- tool: `tools/intel-qwen36-gpu-conv-history-state-capture-probe.py`
- schema: `intel-qwen36-gpu-conv-history-state-capture-probe-v0`
- OpenCL source: `engine/gpu/opencl/q4x8_matvec.cl`
- OpenCL source SHA256:
  `66d8726ca3e80e5decb27c29b186943471700ebe88b8644e903f6932bb189fce`
- capture source: llama.cpp `llm_build_delta_net_base::build_conv_state`,
  tensor `conv_states-5`, before `conv_input` concat
- selected device: `Intel(R) Arc(TM) B390 GPU`
- required checks passed
- captured conv history state: `24576` F32 values, `98304` bytes,
  SHA256
  `61c8b08f1ce5ca53dbbba496efb3262523e56353ced9a601f329626d4167ad36`
- captured `conv_states_reshaped-5` has the same payload SHA256
- GPU QKV vs oracle max abs/RMSE:
  `4.768371582e-06` / `2.961922258e-07`
- CPU conv with captured state vs oracle `conv_output_raw-5` max abs/RMSE:
  `4.768371582e-07` / `1.592260878e-08`
- GPU conv with captured state vs oracle `conv_output_raw-5` max abs/RMSE:
  `1.430511475e-06` / `3.626227524e-08`
- GPU next conv state vs CPU captured-state path max abs/RMSE:
  `5.722045898e-06` / `1.797046569e-07`
- timing: QKV kernel min `171.979 us`; conv kernel min `5.625 us`
- note: this closes the missing real pre-token conv-history boundary for layer
  5. The captured `conv_output_raw-5` lane is now a required oracle check when
  driven by captured `conv_states-5`, not a diagnostic against CPU replay. This
  remains captured single-layer evidence only; it does not prove prompt/token
  decode or model throughput.

GPU resident preconv-to-layer shell handoff gate:

- artifact: `output/gpu-resident-preconv-layer-shell-probe-20260629T211206Z/`
- tool: `tools/intel-qwen36-gpu-resident-preconv-layer-shell-probe.py`
- schema: `intel-qwen36-gpu-resident-preconv-layer-shell-probe-v0`
- source state: `output/gpu-conv-history-state-capture-probe-20260629T205556Z/`
- selected device: `Intel(R) Arc(TM) B390 GPU`
- resident API: `preconv_to_layer_shell_load_once_run_many`, load count `1`,
  invocations `5`
- OpenCL source SHA256:
  `66d8726ca3e80e5decb27c29b186943471700ebe88b8644e903f6932bb189fce`
- embedded OpenCL source SHA256:
  `cb6b518694e13b95887ca79bf17019bc3372222cfbd0ab1bfbb6e6ce708ef89a`
- required checks passed
- boundary: starts from captured `attn_norm-5` and captured `conv_states-5`;
  computes GPU QKV, alpha, beta, `z`, F32 conv, postconv prep, delta recurrent,
  output projection, post-attention residual, FFN RMSNorm, selected/shared FFN
  branches, tail aggregation, and residual to `l_out`
- explicit bridges remain visible: captured conv-history input, preconv Q8 host
  bridge, delta-to-attention host boundary, attention output projection Q8
  bridge, attention-front Q4/F32 host boundary, selected-down Q8 bridge, and
  shared-down Q8 bridge
- GPU QKV vs oracle max abs/RMSE:
  `4.768371582e-06` / `2.961922258e-07`
- GPU conv raw vs oracle max abs/RMSE:
  `1.430511475e-06` / `3.626227524e-08`
- GPU final output vs oracle max abs/RMSE:
  `6.705522537e-08` / `5.031033018e-09`
- GPU layer output vs oracle max abs/RMSE:
  `5.960464478e-08` / `9.856554492e-09`
- timing: preconv-to-postconv kernel sum min `609.059 us`; resident
  preconv-to-layer kernel sum min `2042.382 us`
- note: this remains captured single-layer evidence only; it does not prove
  prompt/token decode or model throughput.

GPU resident layer-input RMSNorm-to-layer shell handoff gate:

- artifact:
  `output/gpu-resident-layer-input-rmsnorm-layer-shell-probe-20260629T212325Z/`
- tool:
  `tools/intel-qwen36-gpu-resident-layer-input-rmsnorm-layer-shell-probe.py`
- schema:
  `intel-qwen36-gpu-resident-layer-input-rmsnorm-layer-shell-probe-v0`
- source state: `output/gpu-conv-history-state-capture-probe-20260629T205556Z/`
- selected device: `Intel(R) Arc(TM) B390 GPU`
- resident API: `layer_input_rmsnorm_to_layer_shell_load_once_run_many`, load
  count `1`, invocations `5`
- required checks passed
- boundary: starts from captured layer residual input plus captured
  `conv_states-5`; GPU computes attention RMSNorm, then runs the closed
  preconv-to-layer shell to `l_out`
- captured `attn_norm-5` remains a required oracle check, not a hidden input
- explicit bridges remain visible: captured conv-history input, preconv Q8 host
  bridge, delta-to-attention host boundary, attention output projection Q8
  bridge, attention-front Q4/F32 host boundary, selected-down Q8 bridge, and
  shared-down Q8 bridge
- GPU `attn_norm` vs oracle max abs/RMSE:
  `1.14440918e-05` / `5.02673477e-07`
- GPU QKV vs oracle max abs/RMSE:
  `9.536743164e-06` / `6.608608494e-07`
- GPU conv raw vs oracle max abs/RMSE:
  `2.861022949e-06` / `9.258948442e-08`
- GPU final output vs oracle max abs/RMSE:
  `2.458691597e-07` / `9.139766273e-09`
- GPU layer output vs oracle max abs/RMSE:
  `5.960464478e-08` / `1.084918042e-08`
- timing: layer-input RMSNorm min `191.458 us`; preconv-to-postconv kernel
  sum min `520.725 us`; resident layer-input-to-layer kernel sum min
  `2010.506 us`
- note: this remains captured single-layer evidence only; it does not prove
  prompt/token decode or model throughput.

GPU layer-6 conv-history state capture handoff gate:

- artifact: `output/gpu-conv-history-state-capture-probe-20260629T212826Z/`
- nested capture artifact:
  `output/gpu-conv-history-state-capture-probe-20260629T212826Z/capture/`
- tool: `tools/intel-qwen36-gpu-conv-history-state-capture-probe.py --layer 6`
- schema: `intel-qwen36-gpu-conv-history-state-capture-probe-v0`
- selected device: `Intel(R) Arc(TM) B390 GPU`
- required checks passed
- captured tensor: `conv_states-6`, `24576` F32 values
- GPU QKV vs oracle max abs/RMSE:
  `3.814697266e-06` / `3.19827477e-07`
- GPU conv with captured state vs oracle `conv_output_raw-6` max abs diff:
  `9.536743164e-07`
- timing: QKV kernel min `187.708 us`; conv kernel min `5.312 us`
- note: this is the required layer-6 pre-token conv-history input boundary for
  the two-linear-layer state-carry shell.

GPU resident two-linear-layer state-carry shell handoff gate:

- artifact: `output/gpu-resident-two-linear-layer-shell-probe-20260629T213811Z/`
- tool: `tools/intel-qwen36-gpu-resident-two-linear-layer-shell-probe.py`
- schema: `intel-qwen36-gpu-resident-two-linear-layer-shell-probe-v0`
- source states:
  `output/gpu-conv-history-state-capture-probe-20260629T205556Z/` and
  `output/gpu-conv-history-state-capture-probe-20260629T212826Z/`
- selected device: `Intel(R) Arc(TM) B390 GPU`
- resident API: `two_linear_layer_state_carry_load_once_run_many`, load count
  `1`, invocations `5`
- required checks passed
- boundary: runs layer 5 and layer 6 with one parameterized resident
  linear-layer shell. Layer 6 residual input is layer 5 GPU output, not captured
  `l_out-5`; captured `l_out-5` remains a required state-carry oracle check.
- GPU layer-5 output vs oracle max abs/RMSE:
  `5.960464478e-08` / `1.084918042e-08`
- GPU layer-6 residual input vs oracle max abs/RMSE:
  `5.960464478e-08` / `1.084918042e-08`
- GPU layer-6 `attn_norm` vs oracle max abs/RMSE:
  `1.907348633e-06` / `3.258450011e-07`
- GPU layer-6 QKV vs oracle max abs/RMSE:
  `4.768371582e-06` / `3.457733523e-07`
- GPU layer-6 conv raw vs oracle max abs/RMSE:
  `9.536743164e-07` / `3.95798655e-08`
- GPU layer-6 final output vs oracle max abs/RMSE:
  `6.705522537e-08` / `7.503174249e-09`
- GPU layer-6 output vs oracle max abs/RMSE:
  `7.450580597e-08` / `1.44792346e-08`
- timing: layer-5 shell sum min `1987.173 us`; layer-6 shell sum min
  `1981.338 us`; two-linear-layer resident sum min `3968.511 us`
- note: this remains captured single-token two-layer evidence only; it does not
  prove prompt/token decode or model throughput.

GPU resident layer-7 full-attention state/input handoff gate:

- artifact:
  `output/gpu-resident-layer7-full-attn-input-handoff-probe-20260629T215528Z/`
- tool:
  `tools/intel-qwen36-gpu-resident-layer7-full-attn-input-handoff-probe.py`
- schema:
  `intel-qwen36-gpu-resident-layer7-full-attn-input-handoff-probe-v0`
- source states:
  `output/gpu-conv-history-state-capture-probe-20260629T205556Z/`,
  `output/gpu-conv-history-state-capture-probe-20260629T212826Z/`, and
  `output/r1-full-attn-all-history-capture-20260627T145615Z/`
- selected device: `Intel(R) Arc(TM) B390 GPU`
- resident API:
  `two_linear_layer_to_full_attention_state_input_load_once_run_many`, load
  count `1`, invocations `5`
- required checks passed
- boundary: runs layer 5 and layer 6 with the resident linear-layer shell,
  carries layer 6 GPU output into layer 7, then computes layer-7 attention
  RMSNorm plus Q/K Q4 projections on GPU. Q/K norm and RoPE remain host
  validation boundaries. V remains CPU/native reference because
  `blk.7.attn_v.weight` is `Q6_K`.
- GPU layer-7 residual input vs oracle max abs/RMSE:
  `7.450580597e-08` / `1.44792346e-08`
- GPU layer-7 `attn_norm` vs oracle max abs/RMSE:
  `1.71661377e-05` / `6.683575466e-07`
- GPU layer-7 `q_full` vs oracle max abs/RMSE:
  `6.675720215e-06` / `1.492466514e-06`
- GPU-derived layer-7 Q RoPE vs oracle max abs/RMSE:
  `3.933906555e-06` / `6.080292635e-07`
- GPU-derived layer-7 K RoPE vs oracle max abs/RMSE:
  `3.814697266e-06` / `7.107343014e-07`
- CPU/native layer-7 V vs oracle max abs/RMSE:
  `1.668930054e-06` / `2.663492407e-07`
- timing: layer-7 RMSNorm min `193.645 us`; Q projection min `185 us`;
  K projection min `91.979 us`; layer-7 full-attn input sum min
  `470.624 us`; two-linear plus layer-7 input sum min `4454.968 us`
- note: this remains captured single-token handoff evidence only; it does not
  prove prompt/token decode or model throughput.

GPU resident layer-7 full-attention core/gate/output handoff gate:

- artifact:
  `output/gpu-resident-layer7-full-attn-core-output-handoff-probe-20260629T221848Z/`
- tool:
  `tools/intel-qwen36-gpu-resident-layer7-full-attn-core-output-handoff-probe.py`
- schema:
  `intel-qwen36-gpu-resident-layer7-full-attn-core-output-handoff-probe-v0`
- selected device: `Intel(R) Arc(TM) B390 GPU`
- resident API:
  `two_linear_layer_to_full_attention_core_output_load_once_run_many`, load
  count `1`, invocations `5`
- required checks passed; `speedup_claims_allowed=false`
- boundary: runs resident layer 5 and layer 6, carries layer 6 GPU output into
  layer 7, computes layer-7 attention RMSNorm, Q/K Q4 projections,
  full-attention core, gate, output projection, residual add, and
  post-attention RMSNorm on GPU. V projection remains a CPU Q6 reference
  boundary and layer-7 FFN remains `q6_down_reference_pending`.
- comparison policy: layer-7 input/QK use the strict component policy
  (`max_abs_diff<=5e-3`, `rmse<=1e-3`, `cosine>=0.99999`); full-attn
  core/output components use the existing all-stateful full-attn policy
  (`max_abs_diff<=1.25e-2`, `rmse<=1e-3`, `cosine>=0.99998`).
- GPU layer-7 `attn_pregate` vs oracle max abs/RMSE:
  `0.01075458527` / `0.0003490113071`
- GPU layer-7 `attn_gated` vs oracle max abs/RMSE:
  `0.0001654624939` / `6.249640471e-06`
- GPU layer-7 `attn_output` vs oracle max abs/RMSE:
  `0.0001972913742` / `2.929201902e-05`
- GPU layer-7 attention residual vs oracle max abs/RMSE:
  `0.0001972913742` / `2.929257888e-05`
- GPU layer-7 post-attention RMSNorm vs oracle max abs/RMSE:
  `0.004746198654` / `0.0009901722081`
- timing: layer-7 full-attn input sum min `466.666 us`; core/gate plus
  output-front sum min `681.666 us`; full layer-7 attention sum min
  `1148.332 us`; two-linear plus layer-7 attention sum min `5105.91 us`
- note: this remains captured single-token layer5/6-to-layer7 attention
  handoff evidence only; it does not prove prompt/token decode or model
  throughput.

GPU resident layer-7 full-attention V Q6 handoff gate:

- artifact:
  `output/gpu-resident-layer7-full-attn-v-q6-handoff-probe-20260629T223352Z/`
- tool:
  `tools/intel-qwen36-gpu-resident-layer7-full-attn-v-q6-handoff-probe.py`
- schema:
  `intel-qwen36-gpu-resident-layer7-full-attn-v-q6-handoff-probe-v0`
- selected device: `Intel(R) Arc(TM) B390 GPU`
- resident API:
  `two_linear_layer_to_full_attention_v_q6_core_output_load_once_run_many`,
  load count `1`, invocations `5`
- required checks passed; `speedup_claims_allowed=false`
- boundary: layer 5 and layer 6 still run as resident GPU linear shells and
  feed layer 7. Layer-7 attention RMSNorm, Q/K Q4 projections, V Q6
  projection, full-attention core, gate, output projection, residual add, and
  post-attention RMSNorm are on GPU. Layer-7 FFN remains
  `q6_down_reference_pending`.
- GPU layer-7 V Q6 projection vs oracle max abs/RMSE:
  `1.907348633e-06` / `2.956694093e-07`
- GPU layer-7 `attn_pregate` vs oracle max abs/RMSE:
  `0.01075458527` / `0.0003490169118`
- GPU layer-7 `attn_gated` vs oracle max abs/RMSE:
  `0.0001657009125` / `6.251396979e-06`
- GPU layer-7 `attn_output` vs oracle max abs/RMSE:
  `0.0001974403858` / `2.929252498e-05`
- GPU layer-7 post-attention RMSNorm vs oracle max abs/RMSE:
  `0.004746079445` / `0.0009901663876`
- timing: V Q6 projection min `158.229 us`; layer-7 full-attn input sum min
  `619.686 us`; core/output sum min `680.936 us`; layer-7 attention sum min
  `1300.622 us`; two-linear plus layer-7 attention sum min `5301.216 us`
- note: this remains captured single-token layer5/6-to-layer7 attention
  handoff evidence only; it does not prove prompt/token decode or model
  throughput.

GPU resident layer-7 FFN Q6 down handoff gate:

- artifact:
  `output/gpu-resident-layer7-ffn-q6-down-handoff-probe-20260629T225016Z/`
- tool:
  `tools/intel-qwen36-gpu-resident-layer7-ffn-q6-down-handoff-probe.py`
- schema:
  `intel-qwen36-gpu-resident-layer7-ffn-q6-down-handoff-probe-v0`
- selected device: `Intel(R) Arc(TM) B390 GPU`
- resident API:
  `two_linear_layer_to_full_attention_v_q6_ffn_q6_down_load_once_run_many`,
  load count `1`, invocations `5`
- required checks passed; `speedup_claims_allowed=false`
- boundary: layer 5 and layer 6 still run as resident GPU linear shells and
  feed layer 7. Layer-7 attention through post-attention RMSNorm remains on
  GPU, but this FFN handoff starts from captured layer-7 post-attention RMSNorm
  to isolate selected/shared Q6 down and the FFN tail.
- selected and shared FFN down tensor types are both `Q6_K`.
- GPU/oracle max abs/RMSE: selected down `7.450580597e-08` /
  `8.886459646e-09`, shared down `3.725290298e-08` /
  `6.124089302e-09`, FFN out `1.862645149e-08` /
  `3.417604439e-09`, derived layer output `5.960464478e-08` /
  `3.795168715e-09`.
- timing: layer-7 attention sum min `1302.809 us`; layer-7 FFN Q6 down
  handoff sum min `785.621 us`; two-linear plus layer-7 attention+FFN sum min
  `6084.548 us`.
- note: this is captured single-token handoff evidence only. It does not yet
  prove live post-attention-norm-to-FFN carry, prompt/token decode, or model
  throughput.

GPU resident integrated layer-7 FFN/l_out handoff gate:

- artifact:
  `output/gpu-resident-layer7-ffn-lout-handoff-probe-20260629T230228Z/`
- tool:
  `tools/intel-qwen36-gpu-resident-layer7-ffn-lout-handoff-probe.py`
- schema:
  `intel-qwen36-gpu-resident-layer7-ffn-lout-handoff-probe-v0`
- selected device: `Intel(R) Arc(TM) B390 GPU`
- resident API:
  `two_linear_layer_to_full_attention_v_q6_live_ffn_lout_load_once_run_many`,
  load count `1`, invocations `5`
- required checks passed; `speedup_claims_allowed=false`
- boundary: layer 5 and layer 6 still run as resident GPU linear shells and
  feed layer 7. Layer-7 FFN now starts from live GPU post-attention RMSNorm,
  and derived `l_out-7` uses the live GPU attention residual.
- policy: original layer-7 attention checks remain unchanged. Live FFN requires
  strict GPU-vs-CPU agreement on the live input, selected/shared down plus FFN
  out plus `l_out` max/RMSE oracle bounds, and final `l_out` full-attn policy.
- GPU-vs-CPU max/RMSE: selected down `8.940696716e-08` /
  `9.662905423e-09`, shared down `4.768371582e-07` /
  `1.759203156e-08`, FFN out `7.823109627e-08` /
  `5.445729275e-09`, derived `l_out` `1.192092896e-07` /
  `5.899565099e-09`.
- GPU-vs-oracle max/RMSE: selected down `0.002783287782` /
  `0.0004185095868`, shared down `0.003078818321` /
  `0.0003550256853`, FFN out `0.0006476454437` /
  `0.000179957983`, derived `l_out` `0.000845015049` /
  `0.0001836701054`.
- timing: layer-7 attention sum min `1293.226 us`; live FFN/l_out sum min
  `792.702 us`; two-linear plus layer-7 attention+FFN sum min `6309.546 us`.
- note: this remains captured single-token layer-output evidence only. It does
  not prove prompt/token decode or model throughput.

GPU resident layer-8 state/input diagnostic:

- artifact:
  `output/gpu-resident-layer8-state-input-handoff-probe-20260629T231515Z/`
- tool:
  `tools/intel-qwen36-gpu-resident-layer8-state-input-handoff-probe.py`
- schema:
  `intel-qwen36-gpu-resident-layer8-state-input-handoff-probe-v0`
- selected device: `Intel(R) Arc(TM) B390 GPU`
- resident API:
  `two_linear_layer_to_full_attention_v_q6_live_ffn_lout_layer8_state_input_load_once_run_many`,
  load count `1`, invocations `5`
- result: not closed; required checks failed because layer-8 oracle drift
  exceeds the current state/input policy. Keep this as diagnostic evidence,
  not a promoted handoff gate.
- boundary tested: layer 5 and layer 6 resident GPU shells feed layer 7;
  layer 7 computes live GPU `l_out-7`; layer 8 consumes that live `l_out-7`
  with captured layer-8 conv state.
- GPU-vs-native on the same live input is strict: layer-8 residual input
  max/RMSE `1.192092896e-07` / `5.899565099e-09`, attn RMSNorm
  `1.907348633e-06` / `1.878425772e-07`, QKV
  `3.814697266e-06` / `4.385198564e-07`, conv raw
  `9.536743164e-07` / `5.580549066e-08`.
- GPU-vs-oracle drift after live `l_out-7`: layer-8 residual input
  `0.000845015049` / `0.0001836701054`, attn RMSNorm
  `0.01848220825` / `0.005073745897`, QKV
  `0.06216526031` / `0.01275710711`, conv raw
  `0.02084478736` / `0.001281389462`.
- timing: layer-8 state/input sum min `696.245 us`; layer5/6/7 plus
  layer-8 state/input sum min `7052.25 us`.
- implication: the handoff machinery is correct relative to native execution
  on the live input, but the accepted layer-7 full-attention/l_out drift is
  too large to keep extending layers under oracle correctness. The next step
  remains layer-8 state/input, focused on reducing or bounding that drift.

Layer-7 full-attention drift diagnostics:

- core variants:
  `output/gpu-layer7-full-attn-core-variant-diagnostic-probe-20260629T233028Z/`;
  required checks passed, diagnostic only.
- baseline replay vs oracle: pregate max/RMSE `0.01077127457` /
  `0.0003491332732`; post-norm `0.004744648933` / `0.0009901930848`;
  derived layer output `0.0008333921432` / `0.0001835386504`.
- FP16 K/V variants reduce core/post-norm drift but do not close the layer
  output: best pregate RMSE `0.0002702061487`, best post-norm RMSE
  `0.0008726595108`, best layer-output RMSE `0.0001778446743`.
- implied weights:
  `output/layer7-full-attn-implied-weight-diagnostic-20260629T233511Z/`;
  required checks passed, diagnostic only.
- floor GQA mapping is validated: F32-V reconstruction RMSE mean/max
  `0.00017455306203473353` / `0.00035138891557609715`, with no negative-weight
  heads. Modulo mapping is rejected by RMSE mean `0.22951159856869174` and
  eight negative-weight heads.
- inferred oracle weights are closer to fp16-K softmax than f32-K softmax
  (mean L1 `0.0008955528616271856` vs `0.0010847121459972439`), but residual
  reconstruction error is still the same order as layer-output drift.
- no-flash capture:
  `output/layer7-full-attn-noflash-capture-diagnostic-20260629T234151Z/`;
  required checks passed, diagnostic only. The temporary target-source edit was
  restored after capture.
- no-flash capture produced `kq_soft_max-7`, confirming that the existing
  flash-attn oracle path hides the softmax weights. It is not a same-input
  replacement for the flash oracle: disabling flash changes upstream
  full-attention outputs before layer 7. Current no-flash vs flash all-history
  Q/K/V max/RMSE are `0.19409632682800293` / `0.03921311927946852`,
  `0.1514371633529663` / `0.04511778966327053`, and
  `0.06792563199996948` / `0.02134677356749859`.
- no-flash layer-7 pregate vs flash oracle pregate is therefore diagnostic
  rather than corrective: max/RMSE `0.21129965782165527` /
  `0.012596365897688758`.
- implication: do not expand past the layer-8 drift gate yet. Next inspect the
  flash-attn numeric contract directly, not a no-flash replacement path.

Layer-7 flash-attn numeric contract follow-up:

- diagnostic:
  `output/gpu-layer7-full-attn-core-variant-diagnostic-probe-20260630T002037Z/`;
  required checks passed, diagnostic only.
- the effective llama.cpp CPU path for this target is AVX/F16C, not AVX512:
  target CPU flags include `avx`, `avx2`, `f16c`, and `fma`, while not listing
  AVX512. `ggml_vec_dot_f16` therefore uses F32 AVX lanes over FP16-loaded Q/K.
- the missing contracts were nearest-even FP16 conversion plus the flash
  one-chunk FP16 V accumulator. Q/K/V are rounded to FP16, online softmax is
  used, `ggml_vec_scale_f16` and `ggml_vec_mad_f16` store the accumulator back
  to FP16 before final F32 normalization, and AVX/F16C dot reduction order is
  needed for exact pregate replay.
- best diagnostic variant:
  `flash_one_chunk_fp16_even_qkv_f16_accum_avx_dot`, pregate max/RMSE `0` /
  `0`; attention output `0.00000005960464478` / `0.000000005100774688`;
  post-attention RMSNorm `0.000001430511475` / `0.0000002320130355`;
  derived layer output `0.0000001192092896` / `0.000000014153822`.
- nearest-even FP16 conversion was the decisive residual fix after the FP16 V
  accumulator; the diagnostic now closes layer-7 flash-attn replay to
  component-level numeric noise.

Resident layer-7 flash-style full-attention core update:

- resident core/output artifact:
  `output/gpu-resident-layer7-full-attn-core-output-handoff-probe-20260630T002318Z/`;
  required checks passed.
- implementation changed the generated OpenCL full-attention core in
  `tools/intel-qwen36-gpu-resident-layer7-full-attn-core-output-handoff-probe.py`
  to use explicit nearest-even FP16 bit round-trip for Q/K/V, AVX/F16C-like dot
  reduction, and the online FP16 V accumulator. Native OpenCL `half` did not
  reproduce the host diagnostic, so the kernel now uses manual helpers.
- resident core/output GPU-vs-oracle: pregate max/RMSE `0.0000004768371582` /
  `0.00000002134382864`; attention output `0.0000002682209015` /
  `0.00000001359233792`; post-attention RMSNorm `0.000003576278687` /
  `0.0000006412779211`.
- V-Q6 integrated artifact:
  `output/gpu-resident-layer7-full-attn-v-q6-handoff-probe-20260630T002434Z/`;
  required checks passed, V projection max/RMSE `0.000001907348633` /
  `0.0000002956694093`, and the attention path stays at the flash-style core
  drift above.
- live FFN/l_out artifact:
  `output/gpu-resident-layer7-ffn-lout-handoff-probe-20260630T002605Z/`;
  required checks passed. Derived live `l_out-7` GPU-vs-oracle max/RMSE is
  `0.0000003576278687` / `0.0000000233569945`, down from the previous
  `0.000845015049` / `0.0001836701054`.

Layer-8 state/input gate after exact flash contract:

- artifact:
  `output/gpu-resident-layer8-state-input-handoff-probe-20260630T002721Z/`;
  required checks passed. This closes the captured single-token layer5/6/7 to
  layer8 state/input handoff gate.
- same-live-input GPU-vs-native remains strict: residual input max/RMSE
  below the strict policy; GPU-vs-oracle is also now below policy: residual
  input max/RMSE `0.0000003576278687` / `0.0000000233569945`; layer-8
  attention RMSNorm `0.000007629394531` / `0.0000006759763424`; QKV
  `0.000006675720215` / `0.0000006227353315`; conv raw
  `0.000001430511475` / `0.00000007979888757`.
- the same artifact also carries the full layer-8 linear layer shell result:
  `probe.checks.layer3.comparisons_passed=true` under the LayerShellResult
  policy, including conv/recurrent state checks. Layer-8 final output
  GPU-vs-oracle max/RMSE is `0.00000009685754776` / `0.00000001015327157`;
  layer output is `0.0000002458691597` / `0.00000004409820403`; linear-attn
  output is `0.000000111758709` / `0.00000002885908591`.
- implication: the layer-8 state/input handoff and full layer shell are correct
  relative to native execution on the live input and to the teacher-forced
  oracle. The next gate can move to layer-9 state/input handoff, still as
  captured single-token evidence.

Layer-9 state/input gate after live layer-8 shell:

- conv-history prerequisite:
  `output/gpu-conv-history-state-capture-probe-20260630T003745Z/`; required
  checks passed for layer 9. QKV GPU-vs-oracle max/RMSE is
  `0.000004768371582` / `0.0000003389489683`; captured-state conv raw
  GPU-vs-oracle max/RMSE is `0.0000009536743164` / `0.00000004734432478`.
- resident handoff artifact:
  `output/gpu-resident-layer9-state-input-handoff-probe-20260630T004635Z/`;
  required checks passed for layers 5/6/7/8/9. This closes the captured
  single-token layer5/6/7/8 to layer9 state/input handoff gate.
- the layer9 reference path feeds live GPU `l_out-8` to both the CPU component
  reference and the GPU shell, avoiding the upstream native-chain drift that
  made the diagnostic
  `output/gpu-resident-layer9-state-input-handoff-probe-20260630T004357Z/`
  fail `layer9_gpu_cpu_matches_native`.
- layer9 state/input GPU-vs-oracle: residual input max/RMSE
  `0.0000002458691597` / `0.00000004409820403`; attention RMSNorm
  `0.000007629394531` / `0.0000009895146415`; QKV
  `0.000004768371582` / `0.0000005900140359`; conv raw
  `0.000001192092896` / `0.00000007806767234`.
- the same artifact also carries the full layer-9 linear layer shell result:
  `probe.checks.layer4.comparisons_passed=true` under the LayerShellResult
  policy, including conv/recurrent state checks. Layer-9 final output
  GPU-vs-oracle max/RMSE is `0.0000001713633537` / `0.00000001198392758`;
  layer output is `0.0000002570450306` / `0.00000004724573521`;
  linear-attn output is `0.00000007450580597` / `0.00000001468654275`.
- implication: the live layer8-to-layer9 handoff and full layer9 shell are
  correct relative to the same live GPU input and teacher-forced oracle. The
  next gate can move to layer-10 state/input handoff, still as captured
  single-token evidence.

Layer-10 Q6 state/input gate after live layer-9 shell:

- diagnostic predecessor:
  `output/gpu-conv-history-state-capture-probe-20260630T005114Z/`; the first
  layer-10 conv-state capture failed because the probe only accepted Q4_K QKV
  while `blk.10.attn_qkv.weight` is Q6_K.
- tensor metadata check: linear-attention layers around this boundary keep
  `blk.10.attn_qkv.weight`, `blk.10.ffn_down_exps.weight`, and
  `blk.10.ffn_down_shexp.weight` as Q6_K; full-attention layers are
  `3, 7, 11, 15, 19, 23, 27, 31, 35, 39`, so layer 11 is the next
  full-attention transition after completing layer 10.
- conv-history prerequisite:
  `output/gpu-conv-history-state-capture-probe-20260630T005924Z/`; required
  checks passed for layer 10 using the Q6_K QKV path. QKV GPU-vs-oracle
  max/RMSE is `0.000003814697266` / `0.0000002470619083`; captured-state conv
  raw GPU-vs-oracle max/RMSE is `0.0000009536743164` / `0.00000003552646665`.
- resident handoff artifact:
  `output/gpu-resident-layer10-state-input-handoff-probe-20260630T011025Z/`;
  required checks passed for layers 5/6/7/8/9/10. This closes the captured
  single-token layer5/6/7/8/9 to layer10 state/input handoff gate.
- the layer10 reference path feeds live GPU `l_out-9` to both the CPU
  component reference and the GPU state/input path. The boundary is explicitly
  `live_gpu_l_out_9`, with captured layer-10 conv state and Q6_K QKV.
- layer10 state/input GPU-vs-oracle: residual input max/RMSE
  `0.0000002570450306` / `0.00000004724573521`; attention RMSNorm
  `0.00001335144043` / `0.000001003427237`; Q6 QKV
  `0.000006675720215` / `0.0000005640600824`; conv raw
  `0.000001430511475` / `0.0000000687826518`. Conv state-after GPU-vs-CPU
  max/RMSE is `0.000003814697266` / `0.0000001188427414`.
- implication: the live layer9-to-layer10 residual/RMSNorm/Q6-QKV/conv
  state-input handoff is correct relative to the same live GPU input and
  teacher-forced oracle. This is not a full layer-10 shell; the next gate must
  add the layer-10 FFN/down side, including Q6_K down tensors, before moving to
  the layer-11 full-attention handoff.

Layer-10 full shell with Q6 QKV and Q6 FFN down:

- new tool:
  `tools/intel-qwen36-gpu-resident-layer10-full-shell-handoff-probe.py`.
  It starts from the closed layer-9 resident shell, extends the reusable
  linear-layer shell to allow Q6_K QKV in `RunGpuPreConvFront`, and keeps the
  existing Q6_K selected/shared down route.
- diagnostic predecessors:
  `output/gpu-resident-layer10-full-shell-handoff-probe-20260630T012440Z/`
  failed remote compile because the generated Q6-QKV buffer used the wrong
  local OpenCL read-write constant; `output/gpu-resident-layer10-full-shell-handoff-probe-20260630T012721Z/`
  passed all numeric comparisons but still had a stale Q4-only selected-down
  shape gate for layer 10.
- passing artifact:
  `output/gpu-resident-layer10-full-shell-handoff-probe-20260630T013038Z/`;
  required checks passed for layers 5/6/7/8/9/10. The artifact requires
  `probe.checks.layer5.comparisons_passed=true`, Q6_K QKV, Q6_K selected down,
  Q6_K shared down, live GPU `l_out-9` as residual input, and captured
  layer-10 conv state.
- layer10 full-shell GPU-vs-oracle: residual input max/RMSE
  `0.0000002570450306` / `0.00000004724573521`; Q6 QKV
  `0.000006675720215` / `0.0000005640600824`; conv raw
  `0.000001430511475` / `0.0000000687826518`; final output
  `0.000000131316483` / `0.00000001126840031`; selected down
  `0.00000006705522537` / `0.000000008831360502`; shared down
  `0.0000001490116119` / `0.00000001561121891`; FFN out
  `0.00000004470348358` / `0.000000005225228126`; layer output
  `0.0000001937150955` / `0.00000004780993139`.
- implication: the live layer9-to-layer10 full linear layer shell is closed
  under the same LayerShellResult policy. The next correctness frontier is the
  layer-11 full-attention handoff from live GPU `l_out-10`.

Layer-11 full-attention handoff from live layer-10 output:

- new tool:
  `tools/intel-qwen36-gpu-resident-layer11-full-attn-handoff-probe.py`.
  It starts from the closed layer-10 full shell, keeps live GPU `l_out-10` as
  the layer-11 residual input boundary, and adds a prefixed layer-11
  full-attention payload loader so layer-7 and layer-11 teacher-forced payloads
  are not conflated.
- diagnostic predecessors:
  `output/gpu-resident-layer11-full-attn-handoff-probe-20260630T014849Z/`
  failed because layer-11 `attn_v.weight` is Q4_K while the first graft reused
  layer-7's V-Q6-only helper;
  `output/gpu-resident-layer11-full-attn-handoff-probe-20260630T015226Z/`
  proved the layer-11 GPU-vs-oracle numeric path but still treated CPU helper
  drift and layer-11 FFN tensor type as required checks.
- passing artifact:
  `output/gpu-resident-layer11-full-attn-handoff-probe-20260630T015834Z/`;
  required checks passed for layers 5/6/7/8/9/10/11. The artifact records
  layer-11 residual input boundary `live_gpu_l_out_10`, V tensor type `Q4_K`,
  V projection boundary `gpu_q4x8_matvec`, and layer-11 FFN boundary
  `q6_down_pending`.
- layer11 full-attention GPU-vs-oracle: residual input max/RMSE
  `0.0000001937150955` / `0.00000004780993139`; attention RMSNorm
  `0.000006675720215` / `0.0000007366104061`; Q full
  `0.000004768371582` / `0.0000008010516637`; Q RoPE
  `0.000003099441528` / `0.0000006698042492`; K RoPE
  `0.000003457069397` / `0.0000007235389919`; V
  `0.000001192092896` / `0.0000002948404135`; pregate
  `0.0004599094391` / `0.00001150628307`; output
  `0.00000009685754776` / `0.0000000232517567`; residual
  `0.0000002086162567` / `0.0000000536195522`; post-attention norm
  `0.000005066394806` / `0.000001201405911`.
- diagnostic timing only: layer11 input `548.644 us`, core/output
  `2434.061 us`, full attention `2982.705 us`. This remains captured
  single-token handoff evidence, not decode throughput.
- implication: layer-11 state/input, Q/K/V, core/gate/output projection,
  attention residual, and post-attention norm are closed against the
  teacher-forced oracle from live GPU `l_out-10`. The follow-up FFN/l_out gate
  must not assume Q6 down tensors.

Layer-11 FFN/l_out handoff from live layer-11 post-attention norm:

- new tool:
  `tools/intel-qwen36-gpu-resident-layer11-ffn-lout-handoff-probe.py`.
  It starts from the closed layer-11 full-attention handoff, keeps live GPU
  layer-11 post-attention norm as the FFN input, keeps the live layer-11
  attention residual as the output residual boundary, and stages prefixed
  layer-11 FFN/l_out oracle payloads.
- passing artifact:
  `output/gpu-resident-layer11-ffn-lout-handoff-probe-20260630T021346Z/`;
  required checks passed for layers 5/6/7/8/9/10/11. The artifact records
  resident API `layer5_to_layer11_ffn_lout_load_once_run_many`, layer-11 FFN
  boundary `gpu_live_post_norm_to_q4_q6_down`, selected down type `Q4_K`,
  shared down type `Q4_K`, and `live_gpu_l_out_11` as the layer-output
  boundary.
- layer11 FFN/l_out GPU-vs-oracle: selected down max/RMSE
  `0.00000008940696716` / `0.00000001719050989`; shared down
  `0.0000002682209015` / `0.00000001582661632`; FFN out
  `0.00000008940696716` / `0.000000008505024139`; layer output
  `0.0000002197921276` / `0.00000005467417474`.
- diagnostic timing only: layer11 full attention `2990.725 us`, FFN/l_out
  `743.637 us`, and layers 5/6/7/8/9/10 to layer11 l_out `17649.287 us`.
  This remains captured single-token handoff evidence, not decode throughput.
- diagnostic note: raw probe check `layer11_ffn_q6_boundary=false` is expected
  because both layer-11 down tensors are `Q4_K`; the required boundary is the
  explicit Q4/Q6 layout check.
- implication: layer-11 FFN and final residual output are now closed from the
  live GPU layer-11 attention state. The next correctness frontier is the
  layer-12 state/input handoff from live GPU `l_out-11`.

Layer-12 state/input handoff from live layer-11 output:

- conv-history prerequisite:
  `output/gpu-conv-history-state-capture-probe-20260630T022043Z/`; required
  checks passed for layer 12. It records layer-12 QKV tensor type `Q4_K`;
  QKV GPU-vs-oracle max/RMSE is `0.000004768371582` /
  `0.0000002886509841`; captured-state conv raw GPU-vs-oracle max/RMSE is
  `0.0000009536743164` / `0.00000004006133205`.
- new tool:
  `tools/intel-qwen36-gpu-resident-layer12-state-input-handoff-probe.py`.
  It starts from the closed layer-11 FFN/l_out handoff, stages layer-12
  payloads under `l12_` to avoid the earlier layer-7 FFN stage-name collision,
  and keeps live GPU `l_out-11` as the layer-12 residual input boundary.
- diagnostic predecessor:
  `output/gpu-resident-layer12-state-input-handoff-probe-20260630T022754Z/`
  failed because actual layer-12 payloads were first staged with `l7_` names,
  overwriting the existing layer-7 FFN payloads used by the upstream
  full-attention gate.
- passing artifact:
  `output/gpu-resident-layer12-state-input-handoff-probe-20260630T023253Z/`;
  required checks passed for layers 5/6/7/8/9/10/11/12. The artifact records
  resident API `layer5_to_layer12_state_input_load_once_run_many`, layer-12
  residual input boundary `live_gpu_l_out_11`, QKV type `Q4_K`, selected down
  type `Q4_K`, shared down type `Q4_K`, and captured layer-12 conv state.
- layer12 state/input GPU-vs-oracle: residual input max/RMSE
  `0.0000002197921276` / `0.00000005467417474`; attention RMSNorm
  `0.00001335144043` / `0.00000101344219`; QKV
  `0.000009536743164` / `0.0000006610689723`; conv raw
  `0.000001907348633` / `0.00000009279256166`.
- the same artifact carries diagnostic full-shell evidence, but top-level
  required checks close only the state/input gate: layer output max/RMSE is
  `0.0000002421438694` / `0.00000005829185102`, and
  `probe.checks.layer12_full_shell_matches_oracle=true`.
- diagnostic timing only: layer12 state/input `688.225 us`, and layers
  5/6/7/8/9/10/11 to layer12 state/input `18681.15 us`. This remains
  captured single-token evidence, not decode throughput.
- implication: layer-12 residual input, attention RMSNorm, Q4 QKV, conv raw,
  conv state-after, and Q4/Q6 tensor layout are closed from live GPU
  `l_out-11`. The next gate should promote the already-passing layer-12 full
  shell/l_out evidence into required checks before moving to layer 13.

Layer-12 full-shell/l_out promotion:

- new tool:
  `tools/intel-qwen36-gpu-resident-layer12-full-shell-handoff-probe.py`.
  It reruns the same target-side layer12 path, changes the resident API label
  to `layer5_to_layer12_full_shell_lout_load_once_run_many`, and promotes the
  already-computed full shell/l_out comparisons into required checks.
- closed artifact:
  `output/gpu-resident-layer12-full-shell-handoff-probe-20260630T024814Z/`;
  required checks passed for layers 5/6/7/8/9/10/11/12, failed checks are
  empty, and `speedup_claims_allowed=false`.
- recorded boundaries: layer-12 residual input `live_gpu_l_out_11`, QKV type
  `Q4_K`, selected/shared down types `Q4_K` / `Q4_K`, captured layer-12 conv
  state, and `layer12_full_shell_matches_oracle=true`.
- layer12 required state/input GPU-vs-oracle remains closed: residual input
  max/RMSE `0.0000002197921276` / `0.00000005467417474`; attention RMSNorm
  `0.00001335144043` / `0.00000101344219`; QKV `0.000009536743164` /
  `0.0000006610689723`; conv raw `0.000001907348633` /
  `0.00000009279256166`.
- layer12 required full-shell GPU-vs-oracle: final output max/RMSE
  `0.0000002011656761` / `0.00000001530897058`; linear-attention out
  `0.00000007450580597` / `0.00000001706922969`; attention output
  `0.00000002235174179` / `0.0000000004247941551`; attention residual
  `0.0000002384185791` / `0.00000005776605755`; post-attn norm
  `0.000005781650543` / `0.000001269918967`; selected down
  `0.000000111758709` / `0.00000001371513971`; shared down
  `0.0000002384185791` / `0.00000001848998282`; FFN out
  `0.00000003259629011` / `0.000000007412026578`; l_out-12
  `0.0000002421438694` / `0.00000005829185102`.
- diagnostic timing only: layer12 state/input `706.455 us`; layers
  5/6/7/8/9/10/11 to layer12 state/input `18557.826 us`. The promotion
  wrapper does not emit a separate full-shell/l_out timing. This remains
  captured single-token evidence, not decode throughput.
- implication: layer-12 full shell/l_out is now a closed required gate from
  live GPU `l_out-11`. The next correctness frontier is layer-13 state/input
  from live GPU `l_out-12`.

Layer-13 state/input handoff from live layer-12 output:

- conv-history prerequisite:
  `output/gpu-conv-history-state-capture-probe-20260630T025433Z/`; required
  checks passed for layer 13. QKV GPU-vs-oracle max/RMSE is
  `0.000003814697266` / `0.0000002424033916`; captured-state conv raw
  GPU-vs-oracle max/RMSE is `0.0000009536743164` /
  `0.00000003834729019`.
- new tool:
  `tools/intel-qwen36-gpu-resident-layer13-state-input-handoff-probe.py`.
  It starts from the closed layer-12 full shell/l_out path, stages actual
  layer-13 payloads under `l13_`, and keeps live GPU `l_out-12` as the
  layer-13 residual input boundary.
- passing artifact:
  `output/gpu-resident-layer13-state-input-handoff-probe-20260630T030028Z/`;
  required checks passed for layers 5/6/7/8/9/10/11/12/13. The artifact
  records resident API `layer5_to_layer13_state_input_load_once_run_many`,
  layer-13 residual input boundary `live_gpu_l_out_12`, QKV type `Q6_K`,
  selected down type `Q6_K`, shared down type `Q6_K`, and captured layer-13
  conv state.
- layer13 state/input GPU-vs-oracle: residual input max/RMSE
  `0.0000002421438694` / `0.00000005829185102`; attention RMSNorm
  `0.000006854534149` / `0.000001171560015`; QKV
  `0.000007629394531` / `0.0000006629572756`; conv raw
  `0.000002145767212` / `0.00000009050174914`.
- diagnostic full-shell note: the same raw probe computes layer-13 full shell
  outputs, but `checks.layer13_full_shell_matches_oracle=false`, so full
  shell/l_out is not closed. The visible `l13_layer_output` GPU-vs-oracle
  max/RMSE is `0.0000002682209015` / `0.00000006384280488`; the next gate must
  diagnose/close the internal full-shell criterion before promotion.
- diagnostic timing only: layer13 state/input `829.058 us`; layers
  5/6/7/8/9/10/11/12 to layer13 state/input `19437.506 us`. This remains
  captured single-token evidence, not decode throughput.
- implication: layer-13 residual input, attention RMSNorm, Q6 QKV, conv raw,
  conv state-after, and Q4/Q6 tensor layout are closed from live GPU
  `l_out-12`. The next correctness frontier is layer-13 full shell/l_out.

Layer-13 full-shell/l_out promotion:

- new tool:
  `tools/intel-qwen36-gpu-resident-layer13-full-shell-handoff-probe.py`.
  It reruns the target-side layer13 path, changes the resident API label to
  `layer5_to_layer13_full_shell_lout_load_once_run_many`, and promotes
  GPU-vs-oracle full-shell/l_out outputs plus state/recurrent GPU-vs-native
  sanity into required checks.
- closed artifact:
  `output/gpu-resident-layer13-full-shell-handoff-probe-20260630T030815Z/`;
  required checks passed, failed checks are empty, and
  `speedup_claims_allowed=false`.
- recorded boundaries: layer-13 residual input `live_gpu_l_out_12`, QKV type
  `Q6_K`, selected/shared down types `Q6_K` / `Q6_K`, captured layer-13 conv
  state, and resident API `layer5_to_layer13_full_shell_lout_load_once_run_many`.
- layer13 required full-shell GPU-vs-oracle: final output max/RMSE
  `0.0000002458691597` / `0.00000001348173282`; linear-attention out
  `0.00000007078051567` / `0.00000001434978566`; attention output
  `0.00000001303851604` / `0.0000000003521372401`; attention residual
  `0.0000002905726433` / `0.00000005941304211`; post-attn norm
  `0.000005543231964` / `0.000001375648395`; selected gate-up
  `0.000001788139343` / `0.0000003228150497`; selected down
  `0.0000002086162567` / `0.00000004048222587`; shared down
  `0.0000006556510925` / `0.00000005412245311`; FFN out
  `0.0000001341104507` / `0.00000001936782697`; l_out-13
  `0.0000002682209015` / `0.00000006384280488`.
- diagnostic note: raw `checks.layer13_full_shell_matches_oracle=false` remains
  recorded because the raw flag also requires CPU-vs-oracle and GPU-vs-CPU
  strict internal FFN lanes. `promotion.json` records 14 diagnostic strict
  misses, all on CPU-component internal FFN lanes; the required GPU-vs-oracle
  full-shell/l_out checks pass.
- state/recurrent sanity: conv state-after GPU-vs-CPU max/RMSE
  `0.000003814697266` / `0.0000001149702043`; recurrent state GPU-vs-CPU
  `0.0000002980232239` / `0.000000002181693011`.
- implication: layer-13 full shell/l_out is closed from live GPU `l_out-12`
  under the explicit promotion criterion. The next correctness frontier is
  layer-14 state/input from live GPU `l_out-13`.

Layer-14 conv-history prerequisite:

- tool: `tools/intel-qwen36-gpu-conv-history-state-capture-probe.py --layer 14`.
- artifact: `output/gpu-conv-history-state-capture-probe-20260630T031335Z/`;
  required checks passed; no required check failed.
- QKV GPU-vs-oracle max/RMSE: `0.000004768371582` /
  `0.0000003323604684`.
- captured conv-state GPU-vs-CPU max/RMSE:
  `0.000004768371582` / `0.0000002156590666`.
- implication: actual layer-14 QKV and captured conv state are staged for the
  next live-GPU layer transition; no layer-13 or layer-12 payload prefix is
  reused.

Layer-14 state/input handoff from live layer-13 output:

- tool: `tools/intel-qwen36-gpu-resident-layer14-state-input-handoff-probe.py`.
  It starts from the closed layer-13 full shell/l_out path, stages actual
  layer-14 conv-history state under `l14_`, and feeds live GPU `l_out-13` into
  layer-14 RMSNorm, Q4 QKV, and F32 conv.
- closed artifact:
  `output/gpu-resident-layer14-state-input-handoff-probe-20260630T032339Z/`;
  required checks passed, no required check failed, and
  `speedup_claims_allowed=false`.
- recorded boundaries: layer-14 residual input `live_gpu_l_out_13`, QKV type
  `Q4_K`, selected/shared down types `Q4_K` / `Q4_K`, captured layer-14 conv
  state, and resident API `layer5_to_layer14_state_input_load_once_run_many`.
- layer14 state/input GPU-vs-oracle: residual input max/RMSE
  `0.0000002682209015` / `0.00000006384280488`; attention RMSNorm
  `0.000007092952728` / `0.000001026697934`; QKV
  `0.000007629394531` / `0.0000004320107564`; conv raw
  `0.000001013278961` / `0.00000006111271629`.
- the same raw probe reports `checks.layer14_full_shell_matches_oracle=true`,
  so the next step can promote full shell/l_out without changing kernels.
- diagnostic timing only: layer14 state/input `700.1 us`; layers
  5/6/7/8/9/10/11/12/13 to layer14 state/input `19965.313 us`. This remains
  component evidence, not decode throughput.

Layer-14 full shell/l_out promotion:

- tool: `tools/intel-qwen36-gpu-resident-layer14-full-shell-handoff-probe.py`.
  It reruns the target-side layer14 path, changes the resident API label to
  `layer5_to_layer14_full_shell_lout_load_once_run_many`, and promotes the
  full shell/l_out comparisons into required checks.
- closed artifact:
  `output/gpu-resident-layer14-full-shell-handoff-probe-20260630T032906Z/`;
  required checks passed, failed checks are empty, promotion gate is
  `layer14_full_shell_lout`, and raw
  `layer14_full_shell_matches_oracle=true`.
- layer14 required full-shell GPU-vs-oracle: final output max/RMSE
  `0.0000002533197403` / `0.00000001189359959`; linear-attention out
  `0.0000001639127731` / `0.00000003321472021`; attention output
  `0.00000001862645149` / `0.0000000004422175117`; attention residual
  `0.0000003352761269` / `0.00000007245144481`; post-attn norm
  `0.000008180737495` / `0.000001401501071`; selected gate-up
  `0.000001072883606` / `0.0000002038427756`; selected down
  `0.0000001043081284` / `0.00000001633859121`; shared down
  `0.0000001788139343` / `0.00000002626732355`; FFN out
  `0.00000003166496754` / `0.000000008649370548`; l_out-14
  `0.0000003501772881` / `0.00000007328653002`.
- state/recurrent sanity: conv state-after GPU-vs-CPU max/RMSE
  `0.000004768371582` / `0.0000002135487017`; recurrent state GPU-vs-CPU
  `0.0000009536743164` / `0.000000002317352488`.
- implication: layer-14 full shell/l_out is closed from live GPU `l_out-13`.
  Layer 15 must be treated as a full-attention layer, not a linear
  conv-history state/input layer.

Layer-15 route discovery and full-attention promotion:

- linear conv-history capture attempt:
  `output/gpu-conv-history-state-capture-probe-20260630T033721Z/` failed
  because the capture had no `linear_attn_qkv_mixed-15` tensor; the dump
  instead exposed `Qcur_full-15`, `Kcur-15`, `Vcur-15`, `attn_pregate-15`,
  `attn_gated-15`, `attn_output-15`, `attn_residual-15`, and
  `attn_post_norm-15`.
- full-attention handoff tool:
  `tools/intel-qwen36-gpu-resident-layer15-full-attn-handoff-probe.py`.
  Closed artifact:
  `output/gpu-resident-layer15-full-attn-handoff-probe-20260630T040145Z/`;
  required checks passed and failed checks are empty. Resident API:
  `layer5_to_layer15_full_attention_load_once_run_many`.
- full-attention GPU-vs-oracle max/RMSE: residual input
  `0.0000003501772881` / `0.00000007328653002`; attn norm
  `0.000005722045898` / `0.0000007880102724`; Q full
  `0.000003814697266` / `0.0000006760636658`; Q ROPE
  `0.000003814697266` / `0.0000005841910671`; K ROPE
  `0.000003814697266` / `0.0000006526523196`; V
  `0.0000007748603821` / `0.0000002148891881`; attention pregate
  `0.0003518760204` / `0.000006074593947`; attention output
  `0.0000003576278687` / `0.00000004089569244`; post-attn norm
  `0.00001084804535` / `0.000001635533785`.
- timing diagnostic only: layer15 full attention kernel-sum min
  `3003.33 us`; through layer15 full attention `22907.496 us`.
  `speedup_claims_allowed=false`.

Layer-15 FFN/l_out promotion:

- tool: `tools/intel-qwen36-gpu-resident-layer15-ffn-lout-handoff-probe.py`.
  Closed artifact:
  `output/gpu-resident-layer15-ffn-lout-handoff-probe-20260630T042436Z/`;
  required checks passed and failed checks are empty. Resident API:
  `layer5_to_layer15_ffn_lout_load_once_run_many`.
- boundary: live GPU `l_out-14` -> layer15 full attention -> live GPU
  post-attn norm -> selected/shared FFN down + tail -> live GPU `l_out-15`.
  Selected and shared down tensors are both `Q4_K`.
- FFN/l_out GPU-vs-oracle max/RMSE: selected down `0.0000001341104507`
  / `0.000000022127585`; shared down `0.000002264976501` /
  `0.00000006850296358`; FFN out `0.0000004321336746` /
  `0.00000001564778861`; l_out-15 `0.0000008344650269` /
  `0.00000008563438215`.
- timing diagnostic only: layer15 full attention `3015.101 us`; layer15
  FFN/l_out `747.805 us`; through layer15 l_out `24011.139 us`.
  `speedup_claims_allowed=false`.

Layer-16 state/input promotion:

- conv-history prerequisite:
  `output/gpu-conv-history-state-capture-probe-20260630T043202Z/`;
  required checks passed and failed checks are empty. The capture has
  `attn_norm`, `linear_attn_qkv_mixed`, `conv_output_raw`, `conv_state`, and
  `conv_state_reshaped`, confirming the linear state/input route.
- tool: `tools/intel-qwen36-gpu-resident-layer16-state-input-handoff-probe.py`.
  Closed artifact:
  `output/gpu-resident-layer16-state-input-handoff-probe-20260630T044009Z/`;
  required checks passed and failed checks are empty. Resident API:
  `layer5_to_layer16_state_input_load_once_run_many`.
- boundary: live GPU `l_out-15` -> layer16 RMSNorm/QKV/conv state input.
  QKV and selected/shared down tensors are `Q6_K`.
- state/input GPU-vs-oracle max/RMSE: residual input `0.0000008344650269`
  / `0.00000008563438215`; attn norm `0.0000114440918` /
  `0.000001352210245`; QKV `0.000003814697266` /
  `0.0000003549186881`; conv output raw `0.000001192092896` /
  `0.00000004892478903`.
- state sanity GPU-vs-CPU max/RMSE: conv state-after `0.000002861022949`
  / `0.0000001078969507`; recurrent state `0.0000006556510925` /
  `0.000000001968965712`.
- timing diagnostic only: layer16 state/input `825.413 us`; through layer16
  state/input `24588.732 us`. `speedup_claims_allowed=false`.

Layer-16 full shell/l_out promotion:

- tool: `tools/intel-qwen36-gpu-resident-layer16-full-shell-handoff-probe.py`.
  Closed artifact:
  `output/gpu-resident-layer16-full-shell-handoff-probe-20260630T044640Z/`;
  required checks passed, failed checks are empty, promotion gate is
  `layer16_full_shell_lout`, and raw `checks.layer11.comparisons_passed=true`.
- full-shell GPU-vs-oracle max/RMSE: final output `0.0000006556510925` /
  `0.00000001504475661`; linear-attention out `0.0000003222376108` /
  `0.00000001965283309`; attention output `0.00000002980232239` /
  `0.0000000005207816477`; attention residual `0.0000004786998034` /
  `0.00000008577055118`; post-attn norm `0.00001201033592` /
  `0.000001677565581`; selected gate-up `0.000001192092896` /
  `0.0000002068089853`; selected down `0.0000001266598701` /
  `0.00000002165834192`; shared down `0.0000001788139343` /
  `0.00000001280042248`; FFN out `0.00000007823109627` /
  `0.000000009461638376`; l_out-16 `0.0000005960464478` /
  `0.00000008665749763`.
- timing diagnostic only: layer16 state/input `829.579 us`; through layer16
  state/input `24818.214 us`. `speedup_claims_allowed=false`.

Layer-17 state/input promotion:

- conv-history prerequisite:
  `output/gpu-conv-history-state-capture-probe-20260630T045333Z/`;
  required checks passed and failed checks are empty. The capture has
  `attn_norm`, `linear_attn_qkv_mixed`, `conv_output_raw`, `conv_state`, and
  `conv_state_reshaped`, confirming the linear state/input route.
- conv-history GPU-vs-oracle max/RMSE: QKV `0.000005722045898` /
  `0.0000003413180253`; captured-state conv output raw
  `0.000002384185791` / `0.00000005479638328`.
- conv-history timing diagnostic only: QKV kernel min `168.854 us`; conv
  kernel min `5.625 us`.
- tool: `tools/intel-qwen36-gpu-resident-layer17-state-input-handoff-probe.py`.
  Closed artifact:
  `output/gpu-resident-layer17-state-input-handoff-probe-20260630T050512Z/`;
  required checks passed and failed checks are empty. Resident API:
  `layer5_to_layer17_state_input_load_once_run_many`.
- boundary: live GPU `l_out-16` -> layer17 RMSNorm/QKV/conv state input.
  QKV and selected/shared down tensors are `Q4_K`.
- state/input GPU-vs-oracle max/RMSE: residual input `0.0000005960464478`
  / `0.00000008665749763`; attn norm `0.0000114440918` /
  `0.000001395346527`; QKV `0.000008583068848` /
  `0.0000006729717036`; conv output raw `0.000002384185791` /
  `0.00000009679716298`.
- state sanity GPU-vs-CPU max/RMSE: conv state-after `0.000004768371582`
  / `0.0000002192371696`; recurrent state `0.0000003576278687` /
  `0.000000002073005595`.
- timing diagnostic only: layer17 state/input `700.309 us`; through layer17
  state/input `25296.443 us`. `speedup_claims_allowed=false`.

Layer-17 full shell/l_out promotion:

- tool: `tools/intel-qwen36-gpu-resident-layer17-full-shell-handoff-probe.py`.
  Closed artifact:
  `output/gpu-resident-layer17-full-shell-handoff-probe-20260630T051115Z/`;
  required checks passed, failed checks are empty, promotion gate is
  `layer17_full_shell_lout`, raw `checks.layer12.comparisons_passed=true`,
  and strict internal diagnostic failure count is `0`.
- full-shell GPU-vs-oracle max/RMSE: final output `0.0000004768371582` /
  `0.00000001583752677`; linear-attention out `0.0000002235174179` /
  `0.00000001998929561`; attention output `0.00000001862645149` /
  `0.0000000003822527821`; attention residual `0.0000004544854164` /
  `0.00000008786488941`; post-attn norm `0.00001120567322` /
  `0.000001741796173`; selected gate-up `0.000001668930054` /
  `0.0000002603532563`; selected down `0.0000001415610313` /
  `0.00000002952211946`; shared down `0.0000001192092896` /
  `0.00000002529262061`; FFN MoE out `0.00000005215406418` /
  `0.000000011700702`; shared expert gated `0.00000001769512892` /
  `0.000000004126733013`; FFN out `0.00000004749745131` /
  `0.00000001216866917`; l_out-17 `0.0000004507601261` /
  `0.00000008972429449`.
- timing diagnostic only: layer17 state/input `694.787 us`; through layer17
  state/input `25240.812 us`. `speedup_claims_allowed=false`.

Layer-18 state/input promotion:

- conv-history prerequisite:
  `output/gpu-conv-history-state-capture-probe-20260630T051904Z/`;
  required checks passed and failed checks are empty. The capture has
  `attn_norm`, `linear_attn_qkv_mixed`, `conv_output_raw`, `conv_state`, and
  `conv_state_reshaped`, confirming the linear state/input route.
- conv-history GPU-vs-oracle max/RMSE: QKV `0.000003814697266` /
  `0.0000003506150691`; captured-state conv output raw
  `0.000001430511475` / `0.00000005520458703`.
- conv-history timing diagnostic only: QKV kernel min `183.854 us`; conv
  kernel min `5.312 us`.
- tool: `tools/intel-qwen36-gpu-resident-layer18-state-input-handoff-probe.py`.
  Closed artifact:
  `output/gpu-resident-layer18-state-input-handoff-probe-20260630T052745Z/`;
  required checks passed and failed checks are empty. Resident API:
  `layer5_to_layer18_state_input_load_once_run_many`.
- boundary: live GPU `l_out-17` -> layer18 RMSNorm/QKV/conv state input.
  QKV and selected/shared down tensors are `Q4_K`.
- state/input GPU-vs-oracle max/RMSE: residual input `0.0000004507601261`
  / `0.00000008972429449`; attn norm `0.0000114440918` /
  `0.000001374695439`; QKV `0.000007629394531` /
  `0.0000006129085629`; conv output raw `0.000002384185791` /
  `0.00000009809010666`.
- state sanity GPU-vs-CPU max/RMSE: conv state-after `0.000004768371582`
  / `0.0000002313835791`; recurrent state `0.0000004768371582` /
  `0.000000001331095046`.
- timing diagnostic only: layer18 state/input `690.1 us`; through layer18
  state/input `25943.724 us`. `speedup_claims_allowed=false`.

Layer-18 full shell/l_out promotion:

- tool: `tools/intel-qwen36-gpu-resident-layer18-full-shell-handoff-probe.py`.
  Closed artifact:
  `output/gpu-resident-layer18-full-shell-handoff-probe-20260630T053422Z/`;
  required checks passed, failed checks are empty, promotion gate is
  `layer18_full_shell_lout`, raw `checks.layer13.comparisons_passed=true`,
  and strict internal diagnostic failure count is `0`.
- full-shell GPU-vs-oracle max/RMSE: final output `0.000000461935997` /
  `0.00000001739368382`; linear-attention out `0.0000002980232239` /
  `0.00000004195906413`; attention output `0.000000009313225746` /
  `0.0000000002372304604`; attention residual `0.0000003911554813` /
  `0.00000009670575965`; post-attn norm `0.000007748603821` /
  `0.000001580564411`; selected gate-up `0.000002145767212` /
  `0.000000360325142`; selected down `0.000000536441803` /
  `0.00000004567710643`; shared down `0.000004053115845` /
  `0.0000001035716583`; FFN MoE out `0.0000001266598701` /
  `0.00000001701439998`; shared expert gated `0.0000007748603821` /
  `0.0000000197207784`; FFN out `0.0000008940696716` /
  `0.00000002769993851`; l_out-18 `0.0000007152557373` /
  `0.0000001005787774`.
- timing diagnostic only: layer18 state/input `690.517 us`; through layer18
  state/input `25844.041 us`. `speedup_claims_allowed=false`.

Layer-19 route discovery and full-attention promotion:

- layer 19 is a full-attention layer, not a linear conv-history state/input
  layer (`FULL_ATTENTION_LAYERS` contains `19`). The layer-19 entry therefore
  starts from the closed live GPU `l_out-18` and uses all-history KV payloads.
- tool: `tools/intel-qwen36-gpu-resident-layer19-full-attn-handoff-probe.py`.
  Closed artifact:
  `output/gpu-resident-layer19-full-attn-handoff-probe-20260630T055651Z/`;
  required checks passed and failed checks are empty. Resident API:
  `layer5_to_layer19_full_attention_load_once_run_many`.
- boundary: live GPU `l_out-18` -> layer19 RMSNorm/Q/K/V projection, ROPE,
  full-attention core/gate, output projection, residual add, and post-attn
  RMSNorm. V tensor is `Q6_K`, V projection boundary is `gpu_q6_raw_matvec`,
  and FFN remains pending at this gate.
- full-attention GPU-vs-oracle max/RMSE: residual input `0.0000007152557373`
  / `0.0000001005787774`; attn norm `0.000009536743164` /
  `0.000000959863932`; q_full `0.000004768371582` /
  `0.0000006597682153`; q_rope `0.000003814697266` /
  `0.0000006197579894`; k_rope `0.000002861022949` /
  `0.0000005866695359`; v `0.0000008344650269` /
  `0.0000001833410353`; attn pregate `0.00002102181315` /
  `0.0000003293087124`; attn output `0.0000002086162567` /
  `0.00000004865217576`; post-attn norm `0.000007688999176` /
  `0.000001769897726`.
- timing diagnostic only: layer19 full-attn input `630.728 us`;
  core/output `2433.852 us`; full attention `3064.58 us`; through layer19
  full attention `28902.993 us`. `speedup_claims_allowed=false`.

Layer-19 FFN/l_out promotion:

- tool: `tools/intel-qwen36-gpu-resident-layer19-ffn-lout-handoff-probe.py`.
  Closed artifact:
  `output/gpu-resident-layer19-ffn-lout-handoff-probe-20260630T060850Z/`;
  required checks passed and failed checks are empty. Resident API:
  `layer5_to_layer19_ffn_lout_load_once_run_many`.
- boundary: live GPU `l_out-18` -> layer19 full attention -> live GPU
  post-attn norm -> selected/shared FFN down + tail -> live GPU `l_out-19`.
  Selected and shared down tensors are both `Q6_K`.
- FFN/l_out GPU-vs-oracle max/RMSE: selected down `0.0000002235174179`
  / `0.00000003309212715`; shared down `0.000002384185791` /
  `0.00000005850169393`; FFN out `0.0000001788139343` /
  `0.00000001364978174`; l_out-19 `0.0000009536743164` /
  `0.0000001134854422`.
- layer output GPU-vs-CPU max/RMSE on the same live input:
  `0.0000002384185791` / `0.000000009018546528`.
- timing diagnostic only: layer19 full attention `3062.289 us`; FFN/l_out
  `791.142 us`; through layer19 l_out `29765.177 us`.
  `speedup_claims_allowed=false`.

Layer-20 conv-history prerequisite and state/input promotion:

- conv-history prerequisite tool:
  `tools/intel-qwen36-gpu-conv-history-state-capture-probe.py --layer 20`.
  Closed artifact:
  `output/gpu-conv-history-state-capture-probe-20260630T061917Z/`;
  required checks passed and failed checks are empty.
- conv-history numeric evidence: QKV GPU-vs-oracle min `168.958 us`,
  max/RMSE `0.000005722045898` / `0.0000004397482458`; conv captured-state
  GPU-vs-oracle min `5.729 us`, max/RMSE `0.000001192092896` /
  `0.00000007085158863`.
- state/input tool:
  `tools/intel-qwen36-gpu-resident-layer20-state-input-handoff-probe.py`.
  Closed artifact:
  `output/gpu-resident-layer20-state-input-handoff-probe-20260630T062826Z/`;
  required checks passed and failed checks are empty. Resident API:
  `layer5_to_layer20_state_input_load_once_run_many`.
- boundary: live GPU `l_out-19` -> layer20 RMSNorm, Q4 QKV, conv raw, and
  captured conv state. Layer20 selected/shared down tensors are both `Q4_K`.
- state/input GPU-vs-oracle max/RMSE: residual input `0.0000009536743164`
  / `0.0000001134854422`; attn norm `0.000007152557373` /
  `0.00000141358923`; QKV mixed `0.000007629394531` /
  `0.000000562268732`; conv output raw `0.00000262260437` /
  `0.00000009091650389`.
- timing diagnostic only: layer20 state/input `692.08 us`; through layer20
  state/input `30726.113 us`. `speedup_claims_allowed=false`.

Layer-20 full-shell/l_out promotion:

- tool:
  `tools/intel-qwen36-gpu-resident-layer20-full-shell-handoff-probe.py`.
  Closed artifact:
  `output/gpu-resident-layer20-full-shell-handoff-probe-20260630T063557Z/`;
  required checks passed, failed checks are empty, and internal strict
  diagnostic failures are `0`. Resident API:
  `layer5_to_layer20_full_shell_lout_load_once_run_many`.
- boundary: live GPU `l_out-19` -> layer20 full linear-attention shell, FFN,
  and live GPU `l_out-20`; layer20 QKV, selected down, and shared down tensors
  are all `Q4_K`.
- full-shell GPU-vs-oracle max/RMSE: final output `0.0000001639127731`
  / `0.00000001229856439`; linear-attn out `0.0000001490116119` /
  `0.00000002916944021`; attn output `0.00000004470348358` /
  `0.000000001387845476`; attn residual `0.0000009536743164` /
  `0.0000001177801919`; post-attn norm `0.000008761882782` /
  `0.000001863726732`; selected down `0.0000001266598701` /
  `0.00000002162831369`; shared down `0.0000008344650269` /
  `0.00000003173081403`; FFN out `0.000000387430191` /
  `0.00000001744084415`; l_out-20 `0.0000005960464478` /
  `0.0000001183418559`.
- state/recurrent GPU-vs-CPU on the same live input: conv state-after
  max/RMSE `0.000004768371582` / `0.0000002729217992`; recurrent state
  max/RMSE `0.0000004768371582` / `0.000000004399439119`.
- timing diagnostic only: layer20 state/input kernel-sum label `693.331 us`;
  through layer20 state/input label `30486.64 us`.
  `speedup_claims_allowed=false`.

Layer-21 conv-history prerequisite and state/input promotion:

- conv-history prerequisite tool:
  `tools/intel-qwen36-gpu-conv-history-state-capture-probe.py --layer 21`.
  Closed artifact:
  `output/gpu-conv-history-state-capture-probe-20260630T064552Z/`;
  required checks passed and failed checks are empty.
- conv-history numeric evidence: QKV GPU-vs-oracle min `170.416 us`,
  max/RMSE `0.000004768371582` / `0.0000005159528968`; conv captured-state
  GPU-vs-oracle min `5.625 us`, max/RMSE `0.000001430511475` /
  `0.00000007334382128`.
- state/input tool:
  `tools/intel-qwen36-gpu-resident-layer21-state-input-handoff-probe.py`.
  Closed artifact:
  `output/gpu-resident-layer21-state-input-handoff-probe-20260630T065538Z/`;
  required checks passed and failed checks are empty. Resident API:
  `layer5_to_layer21_state_input_load_once_run_many`.
- boundary: live GPU `l_out-20` -> layer21 RMSNorm, Q4 QKV, conv raw, and
  captured conv state. Layer21 selected/shared down tensors are both `Q4_K`.
- state/input GPU-vs-oracle max/RMSE: residual input `0.0000005960464478`
  / `0.0000001183418559`; attn norm `0.000007629394531` /
  `0.000001361502046`; QKV mixed `0.000007629394531` /
  `0.000000622702368`; conv output raw `0.000001430511475` /
  `0.00000008238906634`.
- timing diagnostic only: layer21 state/input `699.684 us`; through layer21
  state/input `31239.028 us`. `speedup_claims_allowed=false`.

Layer-21 full-shell/l_out promotion:

- tool:
  `tools/intel-qwen36-gpu-resident-layer21-full-shell-handoff-probe.py`.
  Closed artifact:
  `output/gpu-resident-layer21-full-shell-handoff-probe-20260630T070242Z/`;
  required checks passed, failed checks are empty, and internal strict
  diagnostic failures are `0`. Resident API:
  `layer5_to_layer21_full_shell_lout_load_once_run_many`.
- boundary: live GPU `l_out-20` -> layer21 full linear-attention shell, FFN,
  and live GPU `l_out-21`; layer21 QKV, selected down, and shared down tensors
  are all `Q4_K`.
- full-shell GPU-vs-oracle max/RMSE: final output `0.0000002458691597`
  / `0.00000001532792184`; linear-attn out `0.0000001788139343` /
  `0.00000002835630276`; attn output `0.0000000111758709` /
  `0.0000000003329446833`; attn residual `0.0000007152557373` /
  `0.0000001218187334`; post-attn norm `0.000008940696716` /
  `0.000001963948556`; selected down `0.0000001490116119` /
  `0.00000002466679883`; shared down `0.00000005960464478` /
  `0.000000009519901007`; FFN out `0.00000005960464478` /
  `0.00000001082100629`; l_out-21 `0.0000008344650269` /
  `0.0000001220453481`.
- state/recurrent GPU-vs-CPU on the same live input: conv state-after
  max/RMSE `0.000005722045898` / `0.0000003113545304`; recurrent state
  max/RMSE `0.0000007748603821` / `0.000000005367045787`.
- timing diagnostic only: layer21 state/input kernel-sum label `692.913 us`;
  through layer21 state/input label `31178.611 us`.
  `speedup_claims_allowed=false`.

Layer-22 conv-history prerequisite and state/input promotion:

- conv-history prerequisite tool:
  `tools/intel-qwen36-gpu-conv-history-state-capture-probe.py --layer 22`.
  Closed artifact:
  `output/gpu-conv-history-state-capture-probe-20260630T071158Z/`;
  required checks passed and failed checks are empty.
- conv-history numeric evidence: QKV GPU-vs-oracle min `339.479 us`,
  max/RMSE `0.000003814697266` / `0.0000002477249839`; conv captured-state
  GPU-vs-oracle min `5.416 us`, max/RMSE `0.0000009536743164` /
  `0.00000003631034915`.
- state/input tool:
  `tools/intel-qwen36-gpu-resident-layer22-state-input-handoff-probe.py`.
  Closed artifact:
  `output/gpu-resident-layer22-state-input-handoff-probe-20260630T072045Z/`;
  required checks passed and failed checks are empty. Resident API:
  `layer5_to_layer22_state_input_load_once_run_many`.
- boundary: live GPU `l_out-21` -> layer22 RMSNorm, Q6 QKV, conv raw, and
  captured conv state. Layer22 selected/shared down tensors are both `Q6_K`.
- state/input GPU-vs-oracle max/RMSE: residual input `0.0000008344650269`
  / `0.0000001220453481`; attn norm `0.000006675720215` /
  `0.000001275351532`; QKV mixed `0.000004768371582` /
  `0.0000002854879422`; conv output raw `0.000001192092896` /
  `0.00000004204220981`.
- state/recurrent GPU-vs-CPU on the same live input: conv state-after
  max/RMSE `0.000003814697266` / `0.0000001106893756`; recurrent state
  max/RMSE `0.0000002384185791` / `0.000000001305797655`.
- timing diagnostic only: layer22 state/input `831.976 us`; through layer22
  state/input `32106.001 us`. `speedup_claims_allowed=false`.

Layer-22 full-shell/l_out promotion:

- tool:
  `tools/intel-qwen36-gpu-resident-layer22-full-shell-handoff-probe.py`.
  Closed artifact:
  `output/gpu-resident-layer22-full-shell-handoff-probe-20260630T073845Z/`;
  required checks passed, failed checks are empty, and promotion gate is
  `layer22_full_shell_lout`. Resident API:
  `layer5_to_layer22_full_shell_lout_load_once_run_many`.
- boundary: live GPU `l_out-21` -> layer22 full linear-attention shell, FFN,
  and live GPU `l_out-22`; layer22 QKV, selected down, and shared down tensors
  are all `Q6_K`.
- full-shell GPU-vs-oracle max/RMSE: final output `0.0000001154839993`
  / `0.00000001264832825`; linear-attn out `0.0001010894775` /
  `0.00001040499335`; attn output `0.000000006519258022` /
  `0.0000000002342653781`; attn residual `0.0001014173031` /
  `0.0000104110639`; post-attn norm `0.001098871231` /
  `0.0001430108921`; selected down `0.001301886979` /
  `0.0002707627324`; shared down `0.002960439771` /
  `0.0005460501718`; FFN out `0.0007608570158` /
  `0.000169978024`; l_out-22 `0.0007725581527` /
  `0.0001708972393`.
- Q6 internal FFN strict cosine checks produced `12` diagnostic misses in
  `promotion.json`, but the live-FFN magnitude policy, final/l_out policy, and
  state/recurrent sanity all passed; keep those strict misses as diagnostics,
  not as a closed-gate blocker.
- state/recurrent GPU-vs-CPU on the same live input: conv state-after
  max/RMSE `0.000003814697266` / `0.0000001106893756`; recurrent state
  max/RMSE `0.0000002384185791` / `0.000000001305797655`.
- timing diagnostic only: layer22 state/input kernel-sum label `829.057 us`;
  through layer22 state/input label `32058.393 us`.
  `speedup_claims_allowed=false`.

Layer-23 full-attention diagnostic, not promoted:

- tool:
  `tools/intel-qwen36-gpu-resident-layer23-full-attn-handoff-probe.py`.
  Diagnostic artifact:
  `output/gpu-resident-layer23-full-attn-handoff-probe-20260630T080404Z/`;
  target-side compile succeeded, the resident API ran, but required checks
  failed. Resident API:
  `layer5_to_layer23_full_attention_load_once_run_many`.
- boundary: closed live GPU `l_out-22` -> layer23 full-attention RMSNorm,
  Q/K/V projection, ROPE, core/gate, output projection, residual add, and
  post-attn RMSNorm. Layer23 V tensor type is `Q4_K`; V projection boundary is
  `gpu_q4x8_matvec`; FFN/l_out remains pending.
- failure mode: GPU equals CPU/native on the same live input for the
  projection-frontier pieces, but the live `l_out-22` drift is amplified by
  layer23 RMSNorm/QKV and exceeds the full-attention oracle policy. Examples:
  residual input GPU-vs-oracle max/RMSE `0.0007725581527` /
  `0.0001708972393`; attn norm `0.006445646286` / `0.001319152366`; Q full
  `0.03275156021` / `0.00737637736`; Q rope `0.06020021439` /
  `0.01504922033`; K rope `0.04914185405` / `0.01248130188`; V
  `0.02351605892` / `0.005882754319`.
- core/output also remain outside policy: attention pregate GPU-vs-oracle
  max/RMSE `0.009206265211` / `0.001533923577`; attention output
  `0.001724109054` / `0.0004305867055`; post-attn norm `0.03074526787` /
  `0.006634885208`.
- same-live-input sanity: attn norm GPU-vs-CPU is exact zero; Q/K/V
  GPU-vs-CPU max errors are at single-float-roundoff scale (`3.814697266e-06`,
  `0.00000274181366`, `0.00000125169754`). This is useful diagnostic evidence,
  but it does not close layer23 full-attention because the oracle policy still
  fails.
- timing diagnostic only: layer23 full-attention input kernel sum `571.144 us`;
  core/output `2434.477 us`; layer23 full-attention `3005.621 us`; through
  layer23 full-attention `35065.895 us`.
  `speedup_claims_allowed=false`.

Layer-23 oracle-input diagnostic:

- tool:
  `tools/intel-qwen36-gpu-resident-layer23-oracle-input-full-attn-diagnostic.py`.
  Diagnostic artifact:
  `output/gpu-resident-layer23-oracle-input-full-attn-diagnostic-20260630T081818Z/`;
  target-side compile succeeded. The original live layer23 checks still fail,
  but the injected oracle-input lane passes strict, full-attention, and
  combined diagnostic checks.
- boundary: captured/oracle layer23 residual input, equivalent to exact
  `l_out-22`, feeds the same GPU layer23 RMSNorm, Q/K/V projection, ROPE,
  core/gate, output projection, residual add, and post-attn RMSNorm kernels.
- oracle-input GPU-vs-oracle max/RMSE: attn norm `0.00001335144043` /
  `0.0000005244971102`; Q full `0.000006675720215` /
  `0.000001327436559`; Q rope `0.000003457069397` /
  `0.00000071919149`; K rope `0.000003814697266` /
  `0.0000007143784681`; V `0.000001311302185` /
  `0.0000003702596677`; attention pregate `0.0000793710351` /
  `0.000001734203015`; attention output `0.0000002682209015` /
  `0.0000000403106018`; post-attn norm `0.000003159046173` /
  `0.00000058133545`.
- conclusion: layer23 kernels are not the active blocker. The live failure is
  inherited from layer22 output drift.

Layer-22 drift source after the oracle-input diagnostic:

- layer22 residual/state-input through QKV and conv remains tight, but
  linear-attention output GPU-vs-oracle is already `0.0001010894775` /
  `0.00001040499335`. Post-attn RMSNorm amplifies this to
  `0.001098871231` / `0.0001430108921`.
- layer22 FFN then carries that live postnorm drift: selected gate/up
  GPU-vs-oracle `0.003092467785` / `0.000769562899`, selected down
  `0.001301886979` / `0.0002707627324`, shared down `0.002960439771` /
  `0.0005460501718`, FFN out `0.0007608570158` / `0.000169978024`, and
  live `l_out-22` `0.0007725581527` / `0.0001708972393`.
- next corrective route: reduce or bypass the layer22 attention/postnorm drift
  before re-running the live layer23 full-attention gate. Do not spend more
  time on layer23 core/output kernels until this source is addressed.

Layer-22 output-projection and same-input sensitivity diagnostics:

- captured output-projection diagnostic:
  `output/gpu-q4x8-output-projection-probe-20260630T083019Z/`.
  The tool now resolves wildcard token-15 payload ordinals and runs both Q4 x8
  variants. Starting from captured/oracle layer22 `final_output`, both variants
  pass: rowlane GPU-vs-oracle max/RMSE `0.0000001080334187` /
  `0.00000002437984922`; group8 GPU-vs-oracle `0.0000001080334187` /
  `0.00000002453282129`. Timing diagnostic only: rowlane `218.541 us`,
  group8 `1412.291 us`. This rules out a bad layer22 `ssm_out.weight` Q4 x8
  kernel on exact input.
- live same-input diagnostic:
  `output/gpu-resident-layer23-oracle-input-full-attn-diagnostic-20260630T084029Z/`.
  It adds layer22-only CPU-on-GPU-input comparisons while preserving the failed
  live layer23 gate and passing layer23 oracle-input diagnostic.
- layer22 conv/final same-input evidence: conv output on live GPU QKV has
  GPU-vs-CPU max/RMSE `0.0000004768371582` / `0.00000001760300943`;
  final-output on live GPU preconv has GPU-vs-CPU `0.00000007450580597` /
  `0.000000003822169732`, while CPU/GPU vs oracle remain at about
  `0.0000001154839993` / `0.00000001264832825`.
- layer22 `ssm_out.weight` same-input evidence: using the live GPU
  `final_output`, CPU matvec and GPU rowlane agree at max/RMSE
  `0.0000001192092896` / `0.00000002524723586`, but both differ from oracle
  at `0.000101` / `0.000010405`. Post-attn RMSNorm is also same-input clean
  (`0.000002861022949` / `0.000000399244697` GPU-vs-CPU) while preserving the
  oracle drift (`0.001098871231` / `0.0001430108921`).
- conclusion: layer22 output projection, residual add, RMSNorm, and delta
  final kernels are same-input clean. The active blocker is sensitivity:
  a tiny live layer22 `final_output` deviation is amplified by `ssm_out.weight`
  and then by post-attn RMSNorm.

Layer-22 oracle final-output bypass diagnostic:

- artifact:
  `output/gpu-resident-layer23-l22-oracle-final-output-bypass-diagnostic-20260630T085354Z/`.
  Required checks pass, and the resident API is
  `layer5_to_layer23_l22_oracle_final_output_bypass_diagnostic`.
- the raw layer22 GPU `final_output` drift is still present:
  GPU-vs-oracle max/RMSE `0.0000001154839993` / `0.00000001264832825`.
  The bypass input itself is exact by construction (`0` / `0`) because it is
  the captured oracle tensor.
- with only that diagnostic input bypassed into layer22 `ssm_out.weight`,
  downstream recovers: layer22 `linear_attn_out` GPU-vs-oracle
  `0.0000001080334187` / `0.00000002437984922`, layer22 post-attn norm
  `0.000009000301361` / `0.000001760552369`, layer22 `l_out`
  `0.0000009536743164` / `0.0000001266407143`, and layer23 residual input
  matches that same `0.0000009536743164` / `0.0000001266407143`.
- layer23 live full-attention then closes under the existing policy:
  `attn_norm` `0.00001525878906` / `0.000001161435311`, `q_full`
  `0.000008106231689` / `0.000001956276484`, `attn_output`
  `0.0000003576278687` / `0.00000004692992436`, and post-attn norm
  `0.000009059906006` / `0.000001843633457`.
- conclusion: the layer23 live failure is not a layer23 kernel bug. It is a
  layer22 source-precision/sensitivity problem before `ssm_out.weight`. The
  next route must reduce the layer22 `final_output` source drift; oracle
  bypass is diagnostic only and is not a backend implementation.

Layer-22 final-output source isolation:

- artifact:
  `output/gpu-resident-layer23-l22-final-output-source-diagnostic-20260630T090622Z/`.
  Required checks pass. The downstream path is still the diagnostic
  oracle-final-output bypass, while the extra comparisons isolate which inputs
  to the layer22 delta/final stage reproduce the raw `final_output` drift.
- variants against captured `final_output`:
  raw live layer22 GPU `final_output` `0.0000001154839993` /
  `0.00000001264832825`; all-oracle delta inputs `0.00000007450580597` /
  `0.000000004882838217`; live GPU Q/K/V with oracle gate/beta/z
  `0.00000008195638657` / `0.000000007615988949`; oracle Q/K/V with live GPU
  gate/beta/z `0.0000001266598701` / `0.00000001107423697`.
- single-modulator split: oracle Q/K/V with live GPU gate gives
  `0.00000007450580597` / `0.000000004878284872`; with live GPU beta gives
  `0.00000004470348358` / `0.000000004802761584`; with live GPU z gives
  `0.0000001229345798` / `0.00000001121029805`.
- conclusion: layer22 `z` projection is the dominant source feeding the tiny
  `final_output` drift. Q/K/V-side drift and gate/beta drift do not reproduce
  the raw final-output drift at the same magnitude.

Layer-22 z-projection single-op diagnostics:

- z-only tool:
  `tools/intel-qwen36-gpu-q4x8-z-projection-probe.py`, created because the
  full preconv fan-in probe cannot run as-is on layer22: `blk.22.attn_qkv.weight`
  is not Q4_K, while the z projection is still tested through the Q4 x8 path.
- rowlane artifact:
  `output/gpu-q4x8-z-projection-probe-20260630T092055Z/`, required true.
  `gpu_vs_cpu` max/RMSE `0.000003814697266` / `0.0000004935386173`;
  `gpu_vs_oracle` `0.000003814697266` / `0.0000004622469102`; min
  `158.125 us`.
- group8 artifact:
  `output/gpu-q4x8-z-projection-probe-20260630T092016Z/`, required true.
  `gpu_vs_cpu` max/RMSE `0.000002861022949` / `0.000000494186356`;
  `gpu_vs_oracle` `0.000002861022949` / `0.0000004668687677`; min
  `1377.916 us`.
- conclusion: group8 lowers max abs slightly but does not reduce RMSE and is
  far slower, so it is not a promising correction route. The next diagnostic
  should test a layer22 z-correction path into delta/final and then non-bypassed
  layer23 full-attention, e.g. native/CPU z or a higher-precision z projection,
  before changing the production backend.

Layer-22 native-z correction diagnostic:

- artifact:
  `output/gpu-resident-layer23-l22-native-z-correction-diagnostic-20260630T092606Z/`.
  Required checks pass. Resident API:
  `layer5_to_layer23_l22_native_z_correction_diagnostic`.
- this run does not use the oracle `final_output` bypass. Layer23 consumes live
  GPU `l_out-22`; only layer22 delta/final receives CPU/native `z` instead of
  GPU Q4 x8 `z`.
- correction input evidence: raw layer22 GPU `z` vs oracle is
  `0.000002861022949` / `0.000000481788955`; native-z correction input is
  `0.000001430511475` / `0.0000002888367773`, with zero GPU-vs-CPU difference
  because the diagnostic feeds the native vector.
- downstream closes: layer22 `final_output` GPU-vs-oracle
  `0.0000002533197403` / `0.00000001041565131`; layer22 `l_out`
  `0.0000009536743164` / `0.0000001280381172`; layer23 residual input matches
  that same `0.0000009536743164` / `0.0000001280381172`.
- non-bypassed layer23 full-attention then passes: `attn_norm`
  `0.00001335144043` / `0.000001107654881`, `q_full`
  `0.000007152557373` / `0.000001705566416`, `attn_output`
  `0.0000002384185791` / `0.00000004131149055`, and post-attn norm
  `0.000009417533875` / `0.000001864262542`.
- conclusion: a better layer22 z projection is sufficient to close the
  non-bypassed layer23 full-attention gate. The native-z path is diagnostic
  only; the production route should replace it with a GPU z correction or
  higher-precision z projection before attempting layer23 FFN/l_out.

Layer-22 GPU z-correction diagnostics:

- rejected F32 route:
  `output/gpu-resident-layer23-l22-gpu-f32-z-correction-diagnostic-20260630T094920Z/`.
  This decodes `attn_gate.weight` to F32 and runs GPU F32 matvec over
  Q8-dequantized RMSNorm input. It improves over raw float-input F32, but still
  fails required checks: z correction input GPU-vs-oracle
  `0.00001287460327` / `0.000001337359618`; layer22 `final_output`
  `0.0000003427267075` / `0.00000002045660732`; `ssm_out.weight` amplifies
  the residual to layer22 `linear_attn_out`
  `0.0001010745764` / `0.00001040504231`, so layer23 full-attention fails.
- first Q4 CPU-order route without FP contraction control:
  `output/gpu-resident-layer23-l22-gpu-q4-cpu-order-z-correction-diagnostic-20260630T100810Z/`.
  The injected GPU Q4_K x Q8_K kernel mirrors CPU row-dot order but still lets
  OpenCL contract multiply-adds; z GPU-vs-CPU remains
  `0.000001013278961` / `0.0000002102916015`, and layer23 still fails. This
  proves the correction is sensitive to exact float operation ordering, not
  only Q4/Q8 format.
- closed Q4 CPU-order route:
  `output/gpu-resident-layer23-l22-gpu-q4-cpu-order-z-correction-diagnostic-20260630T101446Z/`.
  Required checks pass. Resident API:
  `layer5_to_layer23_l22_gpu_q4_cpu_order_z_correction_diagnostic`.
  This run keeps layer23 on live GPU `l_out-22`; no CPU/native z and no
  `final_output` oracle bypass are substituted. The z-only kernel disables
  OpenCL FP contraction and matches CPU z exactly: correction input
  GPU-vs-CPU `0` / `0`; GPU-vs-oracle
  `0.000001430511475` / `0.0000002888367773`.
- downstream matches the native-z diagnostic: layer22 `final_output`
  `0.0000002533197403` / `0.00000001041565131`; layer22
  `linear_attn_out` `0.0000001080334187` / `0.00000002803612918`;
  layer22 `l_out` `0.0000009536743164` / `0.0000001280381172`; layer23
  residual input carries the same `0.0000009536743164` /
  `0.0000001280381172`.
- non-bypassed layer23 full-attention passes with live layer22 output:
  `attn_norm` `0.00001335144043` / `0.000001107654881`, `q_full`
  `0.000007152557373` / `0.000001705566416`, `attn_output`
  `0.0000002384185791` / `0.00000004131149055`, and post-attn norm
  `0.000009417533875` / `0.000001864262542`.
- conclusion: the production candidate is a GPU Q4_K x Q8_K CPU-order
  layer22 z projection with FP contraction disabled before delta/final. The
  next implementation step is to move this out of the diagnostic wrapper and
  into the backend path, then continue from closed layer23 full-attention to
  layer23 FFN/l_out.

Layer-22 Q4 CPU-order z backend promotion:

- implementation:
  `engine/include/intel_qwen36/gpu_q4x8_matvec.hpp`,
  `engine/src/gpu_q4_cpu_order_matvec.cpp`, `engine/CMakeLists.txt`, and the
  layer23 remote compile/staging path now expose and link
  `iq36::RunQ4KCpuOrderMatvec`.
- source-backed artifact:
  `output/gpu-resident-layer23-l22-gpu-q4-cpu-order-z-correction-diagnostic-20260630T104243Z/`.
  Compile and run return codes are both `0`; required checks pass.
- generated probe now calls the engine API, not an injected local helper. The
  summary states the z input is corrected through the engine GPU Q4_K x Q8_K
  CPU-order matvec path.
- correction and downstream evidence are unchanged from the successful helper
  diagnostic: z correction input GPU-vs-CPU `0` / `0`, GPU-vs-oracle
  `0.000001430511475` / `0.0000002888367773`; layer22 `l_out`
  `0.0000009536743164` / `0.0000001280381172`; layer23 full-attention input,
  component policy, and core/output checks all pass.
- conclusion: the native-z replacement is now a reusable backend source path.
  The next gate is layer23 FFN/l_out using this backend z correction, not
  another layer23 full-attention-only rerun.

## Next Implementation Gate

Do not port the whole microbench monolith. The next code step is to continue
from the closed non-bypassed live layer23 full-attention gate into layer23
FFN/l_out while keeping the backend Q4 CPU-order layer22 z correction active:

- start from
  `output/gpu-resident-layer22-full-shell-handoff-probe-20260630T073845Z/`
  and feed live GPU `l_out-22` to layer 23 full-attention RMSNorm, Q/K/V
  projection, ROPE, core/gate, output projection, residual add, and post-attn
  RMSNorm
- layer 23 is in `FULL_ATTENTION_LAYERS`; do not route it through the linear
  conv-history state/input handoff path
- keep actual layer-23 payloads under a distinct `l23_` stage prefix; do not
  reuse earlier names because earlier handoff payloads already own those
  prefixes
- use all-history KV payloads for the full-attention core; verify the V tensor
  type and Q4/Q6 packed layout in the probe instead of assuming it from earlier
  layers
- keep the full-attention input, core/gate, output projection, and post-attn
  residual/norm closed without oracle or native CPU correction while adding
  layer-23 FFN/l_out
- do not regress to the old Q4 x8 z projection for layer22; use the backend
  `iq36::RunQ4KCpuOrderMatvec` path before `ssm_out.weight`
- keep it a captured single-token
  layer5/6/7/8/9/10/11/12/13/14/15/16/17/18/19/20/21/22/23 handoff, not a prompt/token
  decode lane
- keep the resident load-once/run-many boundary; do not regress to one-off
  process-per-probe semantics
- keep Q4/Q6 packed layout selection explicit; plane stream evidence is not
  enough for final layer-transition kernels
- keep CPU/native correctness as the denominator; no throughput claim before
  prompt/token evidence, top-k/logit evidence, and ladder smoothness evidence

## Decision

Proceed with GPU bring-up from the intel-plk OpenCL corpus. The runtime gate is
closed, the raw-vs-plane-repacked stream gate is recorded, and the Q4
llama-style x8 packed stream, shim-backed qmatvec profile, and first Q4 QKV
consumer plus preconv fan-in, preconv-to-conv, postconv prep, delta recurrent,
output projection, FFN/MoE router, selected expert gate-up, selected expert
SwiGLU, selected expert down, and selected MoE weighted aggregation handoff
plus shared expert gate, final FFN output add, post-FFN residual add, and
captured layer shell, resident captured-layer shell, and resident
  selected-expert, shared-expert, and attention-to-FFN layer shell handoff gates
  plus resident postconv-to-layer shell handoff and conv-history state capture
  handoff plus resident preconv-to-layer and layer-input RMSNorm-to-layer shell
  handoff plus resident two-linear-layer state-carry shell handoff and resident
  layer-7 full-attention state/input, exact flash-style core/gate/output, V
  Q6, FFN Q6 down, and integrated FFN/l_out handoffs are closed. The layer-8
  and layer-9 state/input handoffs and full layer shells are now closed as
  well; the layer-10 Q6 state/input and full shell handoffs are closed, the
  layer-11 full-attention plus FFN/l_out handoffs from live `l_out-10` are
  closed, the layer-12 state/input plus full shell/l_out handoffs from live
  `l_out-11` are closed, the layer-13 state/input plus full shell/l_out
  handoffs from live `l_out-12` are closed, the layer-14 state/input plus full
  shell/l_out handoffs from live `l_out-13` are closed, the layer-15
  full-attention plus FFN/l_out handoffs from live `l_out-14` are closed, and
  the layer-16 state/input plus full shell/l_out handoffs from live `l_out-15`
  are closed, and the layer-17 state/input plus full shell/l_out handoffs from
  live `l_out-16` are closed, the layer-18 state/input plus full shell/l_out
  handoffs from live `l_out-17` are closed, and the layer-19 full-attention
  plus FFN/l_out handoffs from live `l_out-18` are closed. The layer-20
  state/input plus full shell/l_out handoffs from live `l_out-19` are closed.
  The layer-21 state/input plus full shell/l_out handoffs from live
  `l_out-20` are closed, and the layer-22 state/input plus full shell/l_out
  handoffs from live `l_out-21` are closed. The layer-23 full-attention probe
  now closes from live GPU `l_out-22` when layer22 z uses the GPU Q4 CPU-order
  correction, and that correction is now available through an engine backend
  source path. The next gate is to close layer-23 FFN/l_out, not a full backend
  port.
