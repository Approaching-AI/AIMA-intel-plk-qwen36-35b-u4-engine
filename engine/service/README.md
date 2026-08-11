# IQ36 resident HTTP service

This directory contains the batch-1, single-model HTTP product surface for the
locked Qwen3.6-35B-A3B OpenVINO U4 carrier. It is compatible with the text
generation subset of the OpenAI API and intentionally rejects unsupported
parameters instead of silently changing their meaning.

## Supported surface

- `GET /v1/models` and `GET /v1/models/{model}`
- `POST /v1/completions`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `GET /v1/responses/{response_id}` and
  `DELETE /v1/responses/{response_id}`
- JSON and SSE streaming responses
- function tools, named/required/automatic tool choice, parallel tool-call
  policy, and multi-turn tool results
- Chat JSON mode and JSON Schema output; output is buffered, parsed, and
  validated before a successful response is exposed
- Responses `previous_response_id` state with bounded serialized bytes, entry
  count, TTL, LRU eviction, retrieval, and deletion
- sampling, deterministic seeds, penalties, stop strings, and log probabilities
- exact-token prefix-state reuse with bounded bytes, entries, TTL, and LRU
- bounded batch-1 queue, disconnect cancellation, request deadlines, graceful
  drain, bearer authentication, health/readiness, Prometheus metrics, and
  structured logs

This is not a claim of compatibility with unrelated OpenAI platform APIs. It
does not implement multimodal input, built-in hosted tools, embeddings,
multi-model loading, background responses, or batch sizes above one. The exact
contract is in
`contracts/qwen36-openai-http-service-contract.json`.

## Context length

Callers submit ordinary text or message histories of any token length that fits
the configured window. They do not select or pad to an engine bucket. The
service tokenizes the complete request and routes it to the smallest accepted
capacity bucket.

The admission rule is:

```text
prompt_tokens + requested_output_tokens <= max_context_length
prompt_tokens <= 131072
requested_output_tokens <= 512
```

`--max-context-length` (or `IQ36_MAX_CONTEXT_LENGTH`) can lower the deployment
limit up to the promoted carrier ceiling. Overflow returns HTTP 400 with
`context_length_exceeded`. Input is never silently truncated; Responses
`truncation=auto` is rejected.

This is a service-level safety and capacity limit, not an engine-bucket field
that callers must provide. Each request supplies ordinary content and its
desired output-token limit; the server performs tokenization and admission.

Internally, aligned prefill chunks and query-one tail calls cover arbitrary
prompt lengths without padding or changing the user's tokens.

## Install

The model and two fingerprinted custom OpenVINO GPU plugins are external
runtime assets and are not Python package data. Build a non-overwriting,
checksum-verified native asset bundle on the bound development target:

```bash
python3 tools/intel-qwen36-package-runtime-assets.py \
  --output /opt/iq36/runtime
```

The bundle contains the two exact plugins, graph helpers, locked model identity
contract, custom CONFIG_FILE and every referenced OpenCL source, plus the
OpenVINO license and third-party notices. Its `manifest.json` records every
file's SHA-256 and a path-scrubbed dynamic-library preflight. It never contains
model weights. Set
`IQ36_REPO_ROOT=/opt/iq36/runtime` and
`IQ36_CUSTOM_CONFIG=/opt/iq36/runtime/openvino/custom/iq36_hot_attention_gqa.xml`.
The runtime independently rejects plugin or CONFIG_FILE fingerprint drift.

This is an asset bundle, not a statically linked appliance. The bound target
must also supply its validated Intel GPU driver/OpenCL loader and normal system
C/C++, TBB, and C runtime libraries. Packaging fails if `ldd` reports an
unresolved plugin dependency on the build target.

The promoted plugin was validated with OpenVINO Runtime
`2026.2.0-21902-90214e5be05-releases/2026/2`, not the later public
`2026.2.0-21903-52ddc073857-releases/2026/2` build. Build the service wheel,
then create a checksum-manifested offline wheelhouse from the exact bound
CPython 3.12 environment:

```bash
SOURCE_DATE_EPOCH=315532800 \
  /home/intel/ov/openvino_env/bin/python -m pip wheel . --no-deps \
  --wheel-dir output/http-service-dist

python3 tools/intel-qwen36-package-python-runtime.py \
  --python /home/intel/ov/openvino_env/bin/python \
  --service-wheel output/http-service-dist/intel_qwen36_server-0.1.1-py3-none-any.whl \
  --output output/http-python-wheelhouse
```

The wheelhouse builder verifies every hashed file in the installed OpenVINO,
GenAI, Tokenizers, and Telemetry `RECORD`, omits generated installation files,
reconstructs the exact wheels with normalized timestamps, downloads the nine
locked generic dependencies, and writes all hashes to `manifest.json`. It
refuses an existing output directory. Install into a new environment without
an index:

```bash
python3.12 -m venv /opt/intel-qwen36/.venv
/opt/intel-qwen36/.venv/bin/python -m pip install \
  --no-index --find-links=output/http-python-wheelhouse pip==26.2.1
/opt/intel-qwen36/.venv/bin/python -m pip install \
  --no-index --find-links=output/http-python-wheelhouse \
  --require-hashes \
  -r output/http-python-wheelhouse/bound-runtime-requirements.txt
/opt/intel-qwen36/.venv/bin/python -m pip check
```

The first offline step upgrades the `pip` bundled by the operating-system
CPython venv. It is part of the security gate even though `pip` is not a
service runtime import; leaving the older seeded installer in a production
environment would make a whole-environment vulnerability audit fail.

Distribution-version pins are backed by a runtime build-string check in both
the controller and isolated workers. Substituting another `2026.2.0` wheel
fails before model hashing, compilation, or readiness.
The wheelhouse additionally emits a hashed, index-resolvable audit input.
`pip-audit --disable-pip --strict` covers those ten distributions without
trying to substitute the three non-index OpenVINO release-candidate builds.
Those three builds and the local service wheel remain covered by exact wheel
hashes, installed `RECORD` verification, source provenance, service tests, and
the repository release scan; they are not misreported as package-index audit
successes.

For the existing development target:

```bash
/home/intel/ov/openvino_env/bin/python tools/intel-qwen36-serve.py \
  --host 127.0.0.1 \
  --port 8000
```

The default startup compiles and prewarms the 2k short profile before opening
the listener. Before loading the tokenizer or worker, it also size- and
SHA-256-verifies all 12 locked text-generation model files (about 19.7 GB) and
checks their aggregate seq2300 fingerprint. `/readyz` therefore means both the
model identity gate and default worker shape warmup completed. The
`--model-verification metadata|off`, `--lazy`, and `--no-prewarm` modes are
diagnostic tradeoffs and are not production-ready settings; readiness reports
the selected verification mode and verified runtime identities.

Use `--help` for all configuration fields. Production path, limit, network,
cache, and timeout settings have the corresponding `IQ36_*` environment forms
shown in `deploy/iq36.env.example`. The diagnostic `--lazy`, `--no-prewarm`,
and unauthenticated-bind override remain explicit CLI-only switches. API keys
should be passed through `IQ36_API_KEY_FILE`, not command-line arguments.

## Requests

The supported generation fields are deliberately explicit:

- Common: `model`, `temperature`, `top_p`, `presence_penalty`,
  `frequency_penalty`, `seed`, `stop`, `stream`, and `user`; local extensions
  are `top_k`, `repetition_penalty`, `ignore_eos`, and `prefix_cache`.
- Chat: text-only `messages`; `max_tokens`/`max_completion_tokens`; `n=1`;
  `logprobs`/`top_logprobs`; `stream_options.include_usage`; function `tools`,
  `tool_choice`, and `parallel_tool_calls`; `response_format`; and the Qwen
  `enable_thinking` switch.
- Completions: one string `prompt`; `max_tokens`; `n=1`; `best_of=1`;
  `suffix=null`; `echo`; integer `logprobs`; and stream usage. `echo` plus
  prompt-token log probabilities is rejected because the carrier does not
  expose prompt logits.
- Responses: text or supported input items; transient `instructions`;
  `max_output_tokens`; function tools; `text.format`; bounded `metadata`;
  `store`; `previous_response_id`; and `truncation=disabled`.

Chat and Responses content is text-only. Built-in hosted tools, external JSON
Schema references, and regex schema keywords are rejected. Unknown or invalid
semantic fields return an OpenAI-shaped 4xx error with a stable `code` instead
of being ignored.

Ordinary text streams emit token deltas as they arrive. Tool-calling,
thinking, structured-output, and stop-string requests are buffered until their
output can be parsed or validated; they still use the same SSE protocol but do
not expose unvalidated or stop-sequence text early.

Chat streaming:

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Authorization: Bearer local-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"qwen3.6-35b-a3b-u4",
    "messages":[{"role":"user","content":"你好"}],
    "temperature":0,
    "max_tokens":64,
    "stream":true,
    "stream_options":{"include_usage":true}
  }'
```

Function tool declaration:

```json
{
  "model": "qwen3.6-35b-a3b-u4",
  "messages": [{"role": "user", "content": "上海天气怎么样？"}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "查询城市天气",
      "strict": true,
      "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": false
      }
    }
  }],
  "tool_choice": "required"
}
```

The service returns tool calls but never executes caller code. Send the result
back as a normal Chat `tool` message, or as a Responses
`function_call_output` item with the returned `call_id`.

Responses state:

```bash
curl http://127.0.0.1:8000/v1/responses \
  -H 'Authorization: Bearer local-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"qwen3.6-35b-a3b-u4",
    "input":"继续刚才的话题",
    "previous_response_id":"resp_...",
    "max_output_tokens":64
  }'
```

`store=false` prevents retrieval and lookup through `previous_response_id`.
Stored response state is process-local, byte/entry bounded, expiring, and lost
on restart. The `instructions` parameter applies only to its current Responses
turn; it is not carried into the next `previous_response_id` turn. A stored
response can be managed with the same methods used by the OpenAI Python SDK:

```bash
curl http://127.0.0.1:8000/v1/responses/resp_... \
  -H 'Authorization: Bearer local-key'
curl -X DELETE http://127.0.0.1:8000/v1/responses/resp_... \
  -H 'Authorization: Bearer local-key'
```

Prefix reuse is enabled by default. Set the local extension
`"prefix_cache": false` on a request to force the resident no-prefix lane.
Usage reports `prompt_tokens_details.cached_tokens` (Chat/Completions) or
`input_tokens_details.cached_tokens` (Responses). Prefix-hit results must not be
reported as cold/no-prefix engine speedups.

## Operations

- `GET /healthz`: controller process is alive.
- `GET /readyz`: backend readiness, model verification result, loaded
  profile/bucket, compile/warmup time, plugin fingerprints, worker PID,
  OpenVINO version, and last backend error.
- `GET /metrics`: Prometheus text format; requires bearer auth when configured.
- Logs are single-line JSON on stderr and contain request metadata, timings,
  and request IDs, but not prompt or generated content.
- Prometheus route labels use fixed templates; attacker-controlled model,
  response, and unknown path components cannot create unbounded series.
- A first request outside the resident bucket/profile can evict and compile a
  different isolated worker. `max_resident_workers=1` is the safe default for
  this 64-GiB target.
- Worker admission requires 8 GiB `MemAvailable` by default. An active request
  is cancelled and its worker terminated if availability drops below 4 GiB.
  A worker that does not acknowledge any cancellation within the configured
  grace period is also terminated and recreated on the next request.

The hardened native deployment example is under `deploy/systemd/`. Terminate
with SIGTERM or SIGINT. The listener stops accepting work, admitted requests
drain until `shutdown_timeout`, then remaining streams are cancelled and all
prefix/response state is cleared.

## Verification

Fast suite:

```bash
PYTHONPATH=engine/service \
  /home/intel/ov/openvino_env/bin/python -m unittest discover \
  -s engine/service/tests -v
```

Release-file and Python runtime dependency scans:

```bash
python3 tools/intel-qwen36-release-audit.py \
  --output output/http-service-security/release-audit.json
python3 -m pip install '.[release]'
pip-audit \
  --requirement output/http-python-wheelhouse/bound-index-requirements.txt \
  --require-hashes --disable-pip --strict --format=json \
  --output output/http-service-security/pip-audit.json
```

The repository scanner covers Git-tracked and non-ignored untracked release
files, rejects high-confidence credential signatures and the private target
hostname, and fails closed if a text file exceeds its scan limit. Binary
runtime artifacts need their own provenance/checksum review; the generated
asset manifest provides that review for the promoted plugins.

The release gate additionally requires real-model short/long carrier, streaming,
tool round-trip, prefix-equivalence, plugin-isolation, memory, and resident
latency evidence. See
`benchmarks/intel-qwen36-35b-a3b-gguf-q4km/http-service-acceptance-matrix.json`.
