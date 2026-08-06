# resident harness adapter

This directory is reserved for the Intel backend adapter behind the
`meta-engine-factory` resident harness contract:

```text
load(model, oracle_bundle)
swap_kernel(boundary_id, impl)
run_boundary(boundary_id)
promote(boundary_id)
```

The current C++ skeleton in `engine/include/intel_qwen36/` defines the contract
surface and rejects placeholder or incomplete oracle bundle paths. The R0 load
artifact `output/r0-resident-harness-load-20260627T061911Z/` runs this current
load path against `oracle/r0-oracle-bundle-20260627T060028Z/` and enters loaded
state after counting 26 token/top-k rows, 26 distribution rows, 524 boundary
input rows, and 524 boundary output rows. This is a harness contract gate, not
an optimized inference runtime or a speed claim.

The minimum oracle bundle directory layout required by `load(model,
oracle_bundle)` is:

- `manifest.json`
- `correctness.json`
- `token-topk-references.jsonl`
- `teacher-forced-distribution-references.jsonl`
- `boundary-references/inputs.jsonl`
- `boundary-references/outputs.jsonl`

Use `tools/intel-qwen36-r0-oracle-bundle-validate.py` before treating such a
directory as R0-close evidence. The harness checks the directory shape; the
validator checks full prompt-ladder and per-boundary coverage.

Run the current load path with:

```sh
python3 tools/intel-qwen36-r0-resident-harness-load.py --bundle-dir oracle/r0-oracle-bundle-20260627T060028Z
```
