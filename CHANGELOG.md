# Changelog

## Unreleased

- Added a first-class performance and correctness report with all 21 paired
  benchmark cases, per-case teacher-forced KLD and top-1 results, exact-token
  counts, context smoothness, jitter, memory, service validation, explicit
  non-claims, and checksum-bound raw-evidence instructions.

## 0.1.0 - 2026-08-06

- Added a resident batch-size-1 HTTP/1.1 inference service with OpenAI-shaped
  Models, Completions, Chat Completions, and Responses endpoints.
- Added JSON and SSE streaming, function tools, validated structured output,
  bounded Responses lifecycle state, bearer authentication, health/readiness,
  Prometheus metrics, request deadlines, cancellation, and graceful drain.
- Added exact-token prefix-state caching with byte, entry, TTL, and LRU bounds.
- Added arbitrary caller context lengths up to the configured product window,
  including the validated 131072-token prompt ceiling, without silent input
  truncation or caller-visible engine buckets.
- Added fail-closed model, plugin, configuration, and OpenVINO runtime identity
  gates plus isolated resident worker processes.
- Added an Apache-2.0 Python service wheel, deterministic offline CPython 3.12
  wheelhouse, native x86_64 runtime bundle, systemd deployment, source-rebuild
  recipes, security policy, release scanning, and acceptance evidence.

The locked model remains an external exact-hash prerequisite and is not
distributed by this repository or release.
