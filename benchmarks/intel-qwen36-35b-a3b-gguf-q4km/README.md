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

These are small specs only. Long prompt payloads must be materialized in an
artifact with exact active-tokenizer counts before any product claim.
