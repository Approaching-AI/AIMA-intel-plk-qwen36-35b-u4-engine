# Changelog

## Unreleased

- Fixed the long-profile LM-head binding for preallocated buffers whose
  physical allocation is larger than the active logical layout. This removes
  the reproduced `v0.1.0` near-bucket failure at 16,380, 32,758, 65,519, and
  131,037 prompt tokens without weakening the contiguous-layout assertions.
- Added a reproducible near-boundary benchmark runner and published controlled
  TTFT, decode throughput, system/process memory, full-vocabulary boundary
  correctness, 67/67 service tests, 18/18 real HTTP smoke, streaming coverage,
  maximum-context validation, and bit-identical source-rebuild evidence.
- Marked the fixed `v0.1.1` fingerprint as a release candidate pending the
  complete 21-case output512 ABBA8 successor and publication gates; the frozen
  `v0.1.0` seq2300 speedup claim is not transferred to it.
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
