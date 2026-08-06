# oracle

Reference bundle staging for `intel-qwen36-35b-a3b-gguf-q4km`.

## Validated bundle (R0 closed)

- bundle: `oracle/r0-oracle-bundle-20260627T060028Z/`
- validation: `output/r0-oracle-bundle-validation-20260627T060238Z/`
- status: `r0_oracle_gate_closed=true`

The machine-readable bundle contract is `oracle/oracle-bundle-contract.json`
(authoritative — this README does not restate its fields).

## Promotion-safe bundle contents

A promotion-safe bundle must include:

- token ids
- top-k logprobs
- teacher-forced distribution references
- per-boundary reference inputs
- per-boundary reference outputs

## Resident harness bundle layout

The resident harness only accepts a bundle directory containing:

- `manifest.json`
- `correctness.json`
- `token-topk-references.jsonl`
- `teacher-forced-distribution-references.jsonl`
- `boundary-references/inputs.jsonl`
- `boundary-references/outputs.jsonl`

This layout is also defined in `oracle/oracle-bundle-contract.json`, which is
authoritative.

## Validate a candidate bundle

```sh
python3 tools/intel-qwen36-r0-oracle-bundle-validate.py --bundle-dir <bundle-dir>
```

Without `--bundle-dir`, the validator scans `oracle/` candidate directories. The
latest validation requires 524 boundary records, 26 prompt rows, full
teacher-forced distribution coverage, and explicit 256k prompt-edge rows.

## Capture pipeline (reference)

The pipeline that produced the validated bundle ran, in order: seed stage →
capture spec → capture queue → runtime preflight → boundary-capture route
preflight → llama.cpp source build route → instrumentation map → patch → build →
run → coverage → fragment assemble → prompt materialize → token-id capture →
top-k smoke → distribution capture → bundle assemble → validate. The per-step
scripts are `tools/intel-qwen36-r0-oracle-*.py`,
`tools/intel-qwen36-r0-boundary-capture-*.py`, and
`tools/intel-qwen36-r0-distribution-capture-*.py`.

The full step-by-step runbook (per-step artifacts, EOS lengths, and the 256k
context-edge findings) is archived at
`doc/frozen/intel-qwen36-35b-a3b-gguf-q4km/oracle-README.archive.md` and in
`meta-log/`; the unavailable-lane decisions are `doc/adr/0001` and
`doc/adr/0002`.
