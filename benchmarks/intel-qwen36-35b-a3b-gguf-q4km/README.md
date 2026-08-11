# intel-qwen36-35b-a3b-gguf-q4km benchmarks

This directory defines the acceptance shape for the locked Intel/Qwen GGUF
workstream.

R0 must refresh the target numbers before any product claim is made. The
bootstrap values are carried from prior same-host Intel work only to seed the
first gate.

Prompt specs for oracle capture, denominator runs, and long-context sentinel
checks live in:

- `prompt-suites.json`
- `prompts/`

Published and maintenance evidence indexes:

- `acceptance-matrix.json` — frozen product-performance and correctness gates
- `http-service-acceptance-matrix.json` — HTTP/release gate state
- `http-near-boundary-regression-2026-08-12.json` — `v0.1.0` arbitrary-length
  incident plus controlled `v0.1.1` release-candidate performance,
  correctness, memory, source-rebuild, and HTTP-smoke results

These are small specs only. Long prompt payloads must be materialized in an
artifact with exact active-tokenizer counts before any product claim.
