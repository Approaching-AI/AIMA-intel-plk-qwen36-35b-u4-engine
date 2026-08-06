# Security policy

## Supported version

Security fixes are applied to the current `main` release candidate. Historical
experiment artifacts and frozen routes are evidence, not supported services.

## Reporting a vulnerability

Use the repository host's private security-advisory channel. Do not open a
public issue for an unpatched vulnerability, credentials, model-license
material, or a prompt/state disclosure. Include the affected commit, endpoint,
configuration, reproduction, impact, and whether the report involves model
state, the custom GPU plugin, or HTTP parsing.

## Deployment boundary

The service is designed for a trusted single-model deployment:

- It binds to loopback by default. A non-loopback bind is rejected unless a
  bearer key is configured or the operator explicitly declares that an
  authenticated reverse proxy owns access control.
- Terminate TLS and apply network rate limits at a reverse proxy. The native
  HTTP listener is HTTP/1.1 and does not terminate TLS.
- Keep the API key in a mode-0600 file or a systemd credential. Secrets in CLI
  arguments, repository files, images, or logs are unsupported.
- Use one security principal per service instance. Prefix snapshots and stored
  Responses histories are shared within that instance; they are not a
  multi-tenant isolation mechanism.
- Prefix snapshots can encode sensitive prompt state. They are memory-only,
  bounded, expiring, evicted by LRU, and cleared on worker eviction/shutdown.
  Disable them per request with `prefix_cache=false` when reuse is undesirable.
- Stored Responses histories are also process-local, bounded, and expiring.
  Use `store=false` for turns that must not be addressable by
  `previous_response_id`.
- Request bytes, context tokens, output tokens, JSON Schema complexity, queue
  depth, request time, cache sizes, and stored-response count are bounded.
  External JSON Schema references are rejected to avoid network retrieval.
- The model is never permitted to execute tools. Tool calls are untrusted model
  output; applications must authorize, validate, sandbox, and audit every tool
  before execution.
- Logs and Prometheus labels omit prompts, outputs, bearer tokens, and tool
  arguments. Restrict access to `/metrics` because operational metadata still
  reveals load and model topology.

Before readiness, the service verifies the exact OpenVINO, GenAI, and
Tokenizers runtime build strings, the aggregate locked-model fingerprint, all 12 model
file sizes and SHA-256 values, the custom GPU plugin, and the CONFIG_FILE. A
matching marketing version is insufficient when its source commit differs.
Do not use metadata-only/off model verification or bypass these checks in
production. Firmware, kernel, driver, OpenVINO, model, or plugin changes define
a new unvalidated product carrier.
