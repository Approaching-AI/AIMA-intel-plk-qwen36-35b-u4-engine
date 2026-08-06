> **ARCHIVED — DO NOT EDIT.** Frozen snapshot of the oracle capture runbook /
> append-only artifact pointers, moved out during the 2026-06-28 distillation.
>
> Live oracle spec → `oracle/README.md` · Machine contract → `oracle/oracle-bundle-contract.json`
> Current state → `doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md` · Timeline → `meta-log/`
> Archived: 2026-06-28

---

# oracle

Reference bundle staging for `intel-qwen36-35b-a3b-gguf-q4km`.

R0 now has a validated full oracle bundle:

- bundle: `oracle/r0-oracle-bundle-20260627T060028Z/`
- validation: `output/r0-oracle-bundle-validation-20260627T060238Z/`
- status: `r0_oracle_gate_closed=true`

A promotion-safe bundle must include:

- token ids
- top-k logprobs
- teacher-forced distribution references
- per-boundary reference inputs
- per-boundary reference outputs

Current evidence:

- `output/r0-cpu-llama-oracle-seed-20260626T033706Z/`
- `output/r0-oracle-seed-stage-20260626T034356Z/`
- `oracle/r0-oracle-bundle-20260627T060028Z/`

Normalize the current seed artifact with:

```sh
python3 tools/intel-qwen36-oracle-seed-stage.py
```

The staged seed contains prompt token ids, first-token top-5 logprobs, and
short greedy generated token ids for the six short/router cases. It explicitly
marks teacher-forced distribution references and per-boundary tensors as
missing.

Stage the per-position short/router distribution seed from the CPU llama.cpp raw
responses with:

```sh
python3 tools/intel-qwen36-teacher-forced-seed-stage.py
```

This records 91 deterministic greedy-reference positions with top-5 logprobs.
It is a teacher-forced distribution seed for short/router cases, not the full
acceptance-ladder bundle.

Replay-check a native candidate JSONL against the staged seed with:

```sh
python3 tools/intel-qwen36-oracle-seed-replay.py --candidate-jsonl <candidate.jsonl>
```

The replay verifier can self-check its schema with:

```sh
python3 tools/intel-qwen36-oracle-seed-replay.py --fixture-from-oracle
```

Do not treat the seed as a complete oracle bundle; use the validated bundle
above for resident harness loading.

The current full-bundle capture contract is:

```sh
python3 tools/intel-qwen36-r0-oracle-capture-spec.py
```

Latest artifact:

- `output/r0-oracle-capture-spec-20260626T072158Z/`

It specifies 17 boundary types, 520 per-layer boundary records, and the
26-prompt acceptance ladder. It is a capture spec only, not a real oracle
bundle and not loadable evidence for R0 resident harness closure.

Assemble the validated full bundle from the captured evidence with:

```sh
python3 tools/intel-qwen36-r0-oracle-bundle-assemble.py
```

Latest bundle:

- `oracle/r0-oracle-bundle-20260627T060028Z/`

Validate the bundle with:

```sh
python3 tools/intel-qwen36-r0-oracle-bundle-validate.py --bundle-dir oracle/r0-oracle-bundle-20260627T060028Z --require-valid-bundle
```

Latest validation:

- `output/r0-oracle-bundle-validation-20260627T060238Z/`

It validates 26 token/top-k rows, 26 teacher-forced distribution rows with
12,744 top-logprob positions, 524 boundary input rows, 524 boundary output
rows, and explicit prompt-edge rows for `sentinel_256k` and
`prefill_shape_256k`.

Check target-side reference runtime capability with:

```sh
python3 tools/intel-qwen36-r0-oracle-runtime-preflight.py
```

Latest preflight:

- `output/r0-oracle-runtime-preflight-20260626T074739Z/`

Result:

- llama.cpp server, tokenizer, locked GGUF, OpenVINO model, and the prior
  llama generation oracle tool are present on target
- teacher-forced distribution capture has a candidate route through the prior
  llama oracle tool's `completion_probabilities` path
- stock llama.cpp/OpenVINO paths do not expose the 17 required per-boundary
  tensors as bundle JSONL
- per-boundary capture therefore needs instrumentation or a reference forward
  path that dumps every queued boundary input/output

Run a bounded current-target distribution capture smoke with:

```sh
python3 tools/intel-qwen36-r0-distribution-capture-smoke.py --max-new-tokens 1
```

Latest smoke:

- `output/r0-distribution-capture-smoke-20260626T075730Z/`

It captured one `short_math_001` llama.cpp `completion_probabilities` row with
one generated token, top-5 logprobs, and `request_status=200`. The row is marked
`smoke_only`; it is not a full teacher-forced distribution bundle.

Capture the bounded short/router distribution subset with:

```sh
python3 tools/intel-qwen36-r0-distribution-capture-short-router.py --max-cases 6
```

Latest short/router capture:

- `output/r0-distribution-capture-short-router-20260626T080938Z/`

It captured six current-target llama.cpp CPU `completion_probabilities` rows
with 429 total top-5 logprob positions. `short_factual_002` and
`short_transform_003` reached EOS before their request limits, so the artifact
records the actual generated lengths, 11 and 18 positions respectively. This is
still only a short/router subset, not the full acceptance-ladder distribution
bundle.

Expand the spec into concrete capture tasks with:

```sh
python3 tools/intel-qwen36-r0-oracle-capture-queue.py
```

Latest queue:

- `output/r0-oracle-capture-queue-20260626T074119Z/`

It contains 1100 required bundle JSONL rows:

- 26 token/top-k tasks for `token-topk-references.jsonl`
- 26 teacher-forced distribution tasks for
  `teacher-forced-distribution-references.jsonl`
- 524 boundary input tensor tasks for `boundary-references/inputs.jsonl`
- 524 boundary output tensor tasks for `boundary-references/outputs.jsonl`

Every queue row is intentionally marked `capture_status=missing`; this queue is
not a bundle.

Preflight the per-boundary capture route with:

```sh
python3 tools/intel-qwen36-r0-boundary-capture-route-preflight.py
```

Latest route preflight:

- `output/r0-boundary-capture-route-preflight-20260627T044738Z/`

It confirms the target llama.cpp install at `/home/intel/llama-cpp/llama-b9518`
is binary-only, no instrumentable llama.cpp source tree is present on the
target, the Intel env user-space `cmake`/`g++`/`ninja` toolchain is available,
and no locked-model `llama-server` process was left running. OpenVINO remains
denominator/sanity evidence, not the locked GGUF boundary source. The selected
next route is to stage exact llama.cpp commit `7c158fbb4` and build with the
Intel env.

Resolve and stage the exact llama.cpp source route with:

```sh
python3 tools/intel-qwen36-r0-llama-source-build-route.py --stage-target-source
```

Latest source route:

- `output/r0-llama-source-build-route-20260627T045059Z/`

It resolves target `version: 9518 (7c158fbb4)` to official upstream commit
`7c158fbb4aec1bdc9c81d6ca0e785139f4826fae` and stages that source tree at
`/home/intel/intel-qwen36-r0/source/llama.cpp-7c158fbb4aec1bdc9c81d6ca0e785139f4826fae`.
The staged tree has `CMakeLists.txt`, `src/llama.cpp`, and `ggml/`, and is ready
for instrumentation. This does not close R0.

Map the staged llama.cpp source to the queued oracle boundary hook points with:

```sh
python3 tools/intel-qwen36-r0-llama-instrumentation-map.py
```

Latest instrumentation map:

- `output/r0-llama-instrumentation-map-20260627T050332Z/`

It maps all 17 required boundary types to line-level hook points in
`src/models/qwen35moe.cpp`, `src/llama-graph.cpp`, and `src/llama-sampler.cpp`.
The target source SHA is
`7c158fbb4aec1bdc9c81d6ca0e785139f4826fae`, and the staged source tree was
clean at capture time. This map does not patch source, build a runtime, dump
tensors, or close R0.

Create and apply the gated boundary-capture instrumentation patch with:

```sh
python3 tools/intel-qwen36-r0-boundary-capture-instrumentation-patch.py --apply-target-patch
```

Latest patch route:

- `output/r0-boundary-capture-instrumentation-patch-20260627T051514Z/`

It patches only the staged target source tree, registering
`tools/qwen36-boundary-capture` in `tools/CMakeLists.txt` and adding the
`llama-qwen36-boundary-capture` executable. The tool decodes one token at a
time with batch size 1 and enables `cb_eval` tensor dumping only at the
configured source token position. After patching, the staged source dirty scope
is exactly `M tools/CMakeLists.txt` and
`?? tools/qwen36-boundary-capture/`. This does not run the model, dump tensors,
or close R0.

Build the boundary-capture executable on the target with:

```sh
python3 tools/intel-qwen36-r0-boundary-capture-build.py
```

Latest build:

- `output/r0-boundary-capture-build-20260627T051710Z/`

The target build directory is
`/home/intel/intel-qwen36-r0/build/llama-qwen36-boundary-capture-20260627T051710Z`,
and the executable is
`/home/intel/intel-qwen36-r0/build/llama-qwen36-boundary-capture-20260627T051710Z/bin/llama-qwen36-boundary-capture`.
Configure, build, and `--help` all returned 0. The build artifact itself does
not run the model; the first locked-model run is recorded below.

Run the first locked-model boundary capture with:

```sh
python3 tools/intel-qwen36-r0-boundary-capture-run.py
```

Latest run:

- `output/r0-boundary-capture-run-20260627T054024Z/`

It ran `llama-qwen36-boundary-capture` on the locked GGUF for
`short_math_001`, source token position 15, with `n_ctx=32`, `ngl=0`, and
batch-size-1 one-token decode steps. This run passed extra filters for the
linear-attention path and layer output tensors. The copied target output
contains:

- `capture-summary.json`
- `sampler-topk.json`
- `tensor-dumps.jsonl`
- 1,493 tensor payload binaries under `payloads/`

The run captured 1,493 tensor JSONL rows, 1,493 payload files, 83,628,864
payload bytes, and logits for the source position. This is raw capture evidence only:
it still must be mapped into `boundary-references/inputs.jsonl` and
`boundary-references/outputs.jsonl` before it can become a full oracle bundle.

Assess raw tensor coverage against the queued boundary tasks with:

```sh
python3 tools/intel-qwen36-r0-boundary-capture-coverage.py
```

Latest coverage:

- `output/r0-boundary-capture-coverage-20260627T054428Z/`

Result:

- input tasks with direct or derived candidates: 434/524
- output tasks with direct candidates: 394/524
- input tasks effective under hybrid policy: 524/524
- output tasks effective under hybrid policy: 524/524
- input all-cues matched: 133/524
- output all-cues matched: 381/524
- route status: `raw_boundary_capture_effectively_covers_queue_with_hybrid_policy`

The hybrid policy records full attention layers as 3, 7, 11, ..., 39 and the
remaining 30 layers as linear-attention layers. Linear-layer RoPE rows are
policy-not-applicable, linear-attention equivalents cover the qkv/attention
rows, and `moe_residual` outputs can be derived from `attn_residual + ffn_out`.
This still does not close R0; a bundle assembler must emit policy-valid JSONL
rows and pass the full oracle bundle validator.

Assemble the boundary-reference fragment with:

```sh
python3 tools/intel-qwen36-r0-boundary-bundle-fragment-assemble.py
```

Latest fragment:

- `output/r0-boundary-bundle-fragment-20260627T054948Z/`

It emits:

- `boundary-references/inputs.jsonl`: 524 rows
- `boundary-references/outputs.jsonl`: 524 rows
- 60 `policy_not_applicable` rows for linear-layer RoPE inputs/outputs
- 40 derived `moe_residual` output payloads computed as
  `attn_residual + ffn_out`

This is a boundary fragment only. It does not include
`token-topk-references.jsonl` or
`teacher-forced-distribution-references.jsonl`, and it does not close the full
oracle bundle gate.

Materialize the queued prompt payloads with the active target tokenizer:

```sh
python3 tools/intel-qwen36-r0-oracle-prompt-materialize.py
```

Latest prompt materialization:

- `output/r0-oracle-prompt-materialization-20260626T082201Z/`

It materialized all 26 prompt rows used by the token/top-k queue. The 20
generated long-context sentinel and prefill rows have exact target
`llama-tokenize` counts through 262144 tokens. This is prompt payload evidence
only; it is not a token/top-k bundle, distribution bundle, or per-boundary
oracle bundle.

Capture full-ladder prompt token IDs from the materialized prompt payloads:

```sh
python3 tools/intel-qwen36-r0-oracle-token-id-capture.py
```

Latest token-id capture:

- `output/r0-oracle-token-id-capture-20260626T083347Z/`

It captured prompt token IDs for all 26 oracle rows, totaling 1,251,478 prompt
tokens with max prompt length 262144. This is token-id evidence only; it does
not include full-ladder top-k logits, distribution references, or boundary
tensors.

Capture a bounded materialized-prompt first-token top-k smoke with:

```sh
python3 tools/intel-qwen36-r0-oracle-topk-smoke.py
```

Latest top-k smoke:

- `output/r0-oracle-topk-smoke-20260626T121946Z/`

It captured first-token top-5 rows for the 128k sentinel/prefill rows
through the current-target llama.cpp CPU server. Earlier smokes covered
`short_math_001`, the 1k sentinel/prefill rows, and the 2k, 4k, 8k, 16k,
32k, 64k, and 100k sentinel/prefill rows:

- `output/r0-oracle-topk-smoke-20260626T084130Z/`
- `output/r0-oracle-topk-smoke-20260626T084753Z/`
- `output/r0-oracle-topk-smoke-20260626T085856Z/`
- `output/r0-oracle-topk-smoke-20260626T092009Z/`
- `output/r0-oracle-topk-smoke-20260626T100409Z/`

Together these prove the materialized long/prefill prompt route through 128k
rows, but this is still not the full-ladder token/top-k bundle.

Latest 256k exact-context attempt:

- `output/r0-oracle-topk-smoke-20260626T144950Z/`

It attempted `sentinel_256k` and `prefill_shape_256k`, both with exact
262144-token prompts and `n_ctx=262144`. Both `/completion` requests returned
HTTP 400 `exceed_context_size_error` with no top-logprobs, while prompt token
counts still matched materialization. This is failure evidence for the current
exact-context llama.cpp route, not a successful top-k capture.

Latest 256k prompt-edge policy:

- `output/r0-oracle-256k-prompt-edge-policy-20260626T145727Z/`
- `doc/adr/0002-r0-256k-prompt-edge-topk-policy.md`

It accepts exact 262144-token first-token top-k as a context-edge unavailable
row for R0. The exact prompts remain valid prompt/token-id evidence, but the
policy does not create top-k logits, does not claim full-ladder token/top-k
coverage, and does not close the oracle gate.

Capture a materialized-prompt distribution subset with:

```sh
python3 tools/intel-qwen36-r0-distribution-capture-materialized.py
```

Latest materialized distribution capture:

- `output/r0-distribution-capture-materialized-20260626T150939Z/`
- `output/r0-distribution-capture-materialized-20260626T151536Z/`
- `output/r0-distribution-capture-materialized-20260626T152722Z/`
- `output/r0-distribution-capture-materialized-20260626T154746Z/`
- `output/r0-distribution-capture-materialized-20260626T165847Z/`
- `output/r0-distribution-capture-materialized-20260626T184541Z/`

Together these capture materialized `sentinel`/`prefill_shape` rows from 1k
through 128k with a 512-token request. The eighteen rows contain 7,154 total top-5
logprob positions. `prefill_shape_001k` stopped at EOS after 29 positions, and
`prefill_shape_008k` stopped at EOS after 420 positions; `sentinel_064k`
stopped at EOS after 15 positions, and `sentinel_100k` stopped at EOS after 17
positions; `sentinel_128k` stopped at EOS after 17 positions. The other thirteen rows
captured all 512 requested positions. This is still a
materialized subset, not the full acceptance distribution bundle.

Latest 1024-token request materialized subsets:

- `output/r0-distribution-capture-materialized-20260626T213023Z/`
- `output/r0-distribution-capture-materialized-20260626T215314Z/`
- `output/r0-distribution-capture-materialized-20260626T221606Z/`
- `output/r0-distribution-capture-materialized-20260626T234204Z/`
- `output/r0-distribution-capture-materialized-20260627T013743Z/`

Together these cover 1k, 2k, 4k, 8k, 16k, 32k, 64k, and 100k
`sentinel`/`prefill_shape` rows, plus the 128k sentinel/prefill rows, with a
1024-token request. The eighteen rows contain 12,315 total top-5 logprob
positions.
`sentinel_001k`, `prefill_shape_001k`, `sentinel_002k`, and `sentinel_004k`
stopped at EOS before the 1024-token request limit; `prefill_shape_002k` and
`prefill_shape_004k` captured all 1024 requested positions. `sentinel_008k`,
`prefill_shape_008k`, and `sentinel_016k` stopped at EOS before the
1024-token request limit; `prefill_shape_016k` captured all 1024 requested
positions. `sentinel_032k` and `sentinel_064k` stopped at EOS before the
1024-token request limit; `prefill_shape_032k` and `prefill_shape_064k`
captured all 1024 requested positions. `sentinel_100k` stopped at EOS before
the 1024-token request limit; `prefill_shape_100k` captured all 1024 requested
positions. `sentinel_128k` stopped at EOS before the 1024-token request limit;
`prefill_shape_128k` captured all 1024 requested positions. This is still a
materialized subset, not the full acceptance distribution bundle.

The resident harness will only accept a real oracle bundle directory containing:

- `manifest.json`
- `correctness.json`
- `token-topk-references.jsonl`
- `teacher-forced-distribution-references.jsonl`
- `boundary-references/inputs.jsonl`
- `boundary-references/outputs.jsonl`

Latest resident harness gate audit:

- structural audit before load:
  `output/r0-resident-harness-gate-audit-20260627T060802Z/`
- load artifact:
  `output/r0-resident-harness-load-20260627T061911Z/`
- post-load gate audit:
  `output/r0-resident-harness-gate-audit-20260627T061917Z/`

The post-load audit records one structurally loadable bundle,
`resident_harness_load_executed=true`, and
`r0_resident_harness_gate_closed=true`. This is the current C++ resident
harness load path; the load artifact also records oracle row counts
26/26/524/524. It is not a speed claim.

Validate whether a candidate directory is a full R0 oracle bundle with:

```sh
python3 tools/intel-qwen36-r0-oracle-bundle-validate.py --bundle-dir <bundle-dir>
```

Without `--bundle-dir`, the validator scans `oracle/` candidate directories.
Latest validation:

- `output/r0-oracle-bundle-validation-20260627T060238Z/`

It requires 524 total boundary records, 26 prompt rows, and full
teacher-forced distribution coverage, with explicit 256k prompt-edge rows
unless a later capture supersedes the policy. The latest scan found one valid
full oracle bundle, so the oracle gate is closed.
