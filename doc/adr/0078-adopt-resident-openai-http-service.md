# ADR 0078: Adopt a resident OpenAI-compatible HTTP service

Date: 2026-08-06

## Status

Accepted

## Context

The promoted OpenVINO engine is currently exposed only through experiment
workers.  Those workers compile a bucket-specific graph, run one bounded row,
write evidence, and exit.  They are not an HTTP service and provide no
streaming protocol, request cancellation, tool-call translation, authentication,
or cross-request prefix reuse.

The product request now requires a release-grade, resident text-inference
surface while preserving the already accepted cold/no-prefix engine result.
The accepted carrier is bucket-scoped: 2k/4k/8k use the seq2291 affine-Q4
full-logit plugin, while 16k/32k/64k/128k greedy timing uses the accepted
legacy compact token-only plugin.  A service must not silently claim that the
compact token-only path supports sampling or log probabilities.

Evidence:

- `output/openvino-affine-q4-product-rollup-20260805Tseq2300-clean/gate.json`
- `tools/intel-qwen36-openvino-product-rollup.py`
- `/home/intel/Qwen3.6-35B-A3B-ov/chat_template.jinja`

## Decision

Add one batch-1, single-model, OpenAI-compatible text service with:

- `/v1/models`, `/v1/completions`, `/v1/chat/completions`, and
  `/v1/responses`;
- JSON and SSE responses, disconnect cancellation, bounded request queuing,
  health/readiness/metrics endpoints, optional bearer authentication, and
  graceful shutdown;
- Qwen's locked tokenizer and chat template, including function-tool prompt
  rendering, multi-turn tool results, and translation of model tool calls to
  OpenAI response objects;
- a bounded, TTL/LRU prefix-state cache whose hits are reported separately
  from the existing cold/no-prefix lane;
- resident, isolated OpenVINO worker processes keyed by frozen
  carrier-profile and context bucket.  The service may retain a configurable
  number of compiled workers, but executes only one request at a time on the
  locked batch-1 target;
- a default full startup identity gate over the 12 locked text-generation
  model files, in addition to the exact OpenVINO/GenAI build identities and
  promoted plugin and CONFIG_FILE hashes;
- arbitrary user prompt lengths up to a configurable total context limit;
  users never need to pad to a product bucket.  The router selects the smallest
  accepted bucket that contains the actual prompt and rejects
  `prompt_tokens + max_new_tokens` overflow without silent truncation;
- capability routing: full-logit carriers for sampling, penalties, or
  log-probability requests; the accepted compact token-only carrier only for
  compatible greedy long-context requests.

This decision does not add multi-model loading, dynamic batching, remote tool
execution, vision/audio input, or unrelated OpenAI platform APIs.  It does not
convert prefix-hit measurements into cold/no-prefix speed claims, and it does
not reopen the accepted seq2300 engine gate.

## Consequences

- The HTTP controller and every OpenVINO worker have separate lifetimes.  A
  worker switch cannot leave the wrong plugin or environment in a carrier.
- Prefix entries contain model state and are single-tenant process-local data;
  they are bounded, expiring, non-persistent, and cleared when their worker is
  evicted.
- Tool definitions are supplied to the model; the server returns tool calls
  but never executes caller code.
- Unsupported OpenAI fields fail with an OpenAI-shaped error instead of being
  ignored when doing so could change generation semantics.
- The Python package pins the release-candidate distribution metadata and the
  process also checks complete runtime build strings. A public wheel with the
  same release family but a different source commit fails before readiness.
- Public release remains gated on the service acceptance matrix, security and
  provenance audit, release documentation, and an explicit repository-license
  decision.

## Follow-Up

- Implement and validate the service contract in
  `contracts/qwen36-openai-http-service-contract.json`.
- Keep service-prefix and cold/no-prefix performance artifacts in distinct
  lanes.
- Supersede this ADR if the runtime gains a proven shared-plugin carrier that
  removes process isolation, or if batch/multi-model scope changes.

> The live gate and next action remain in
> `doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
