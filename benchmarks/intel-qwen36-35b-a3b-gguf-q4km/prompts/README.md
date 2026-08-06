# Prompt Suites

These files define stable inputs for acceptance and profiling. They are specs,
not large generated prompt payloads. Runners should materialize long prompts on
the target using the declared generator fields and the active tokenizer, then
record the actual token counts in the artifact.

Suites:

- `deterministic-greedy.jsonl`: short token-equivalence prompts.
- `router-stability.jsonl`: MoE/router-sensitive token-equivalence prompts.
- `long-context-sentinels.jsonl`: one sentinel retrieval case per required
  context bucket.
- `prefill-shape.jsonl`: one cold no-prefix prefill-shape case per required
  context bucket.

Validate the suite before using it:

```text
python3 tools/validate_repo.py
```

Materialize the current oracle prompt queue with the active target
`llama-tokenize` command:

```text
python3 tools/intel-qwen36-r0-oracle-prompt-materialize.py
```

Latest artifact:

- `output/r0-oracle-prompt-materialization-20260626T082201Z/`

That artifact verifies exact active-tokenizer counts for all generated
sentinel and prefill buckets through 262144 tokens. It is prompt payload
evidence only; full R0 still requires token/top-k references,
teacher-forced distribution references, and per-boundary tensors.
