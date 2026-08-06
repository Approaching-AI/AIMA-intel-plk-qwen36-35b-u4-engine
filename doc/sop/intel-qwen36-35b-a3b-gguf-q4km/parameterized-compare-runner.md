# SOP — parameterized compare runner (collapse 35 drivers → 1)

Snapshot: 2026-06-29

The 35 `tools/intel-qwen36-r1-engine-<boundary>-compare.py` drivers are ~18,000
lines, ~45% byte-identical boilerplate, and they hand-maintain a registry that
already drifted (see Drift below). This is the collapse-ritual violation: one
new file per kernel variant. The target is one runner driven by
`engine/boundaries.json`, so a new boundary = one JSON block + one `.cpp`, with
no new driver.

This is a **target-validated** refactor: it cannot be tested on this host (needs
the PTL ssh/scp/g++ env). Do the migration, then run the parity gate on the
target before deleting any old driver.

## Drift to fix first

`tools/intel-qwen36-r1-engine-layer1-postconv-compare.py` is an orphan: it is in
the `validate_repo.py` hardcoded driver list but **not** in `boundaries.json`,
CMake, or `engine/tests/` (no `.cpp`). Decide: fold into the registry or delete.
This must be resolved before collapse so the boundary set has one definition.

## Common skeleton (identical across all drivers)

`parse_args` (same 7 flags: host/model/env-script/remote-root/oracle-bundle/
out-dir/timeout) → stamp out_dir/remote_dir → resolve oracle reference rows →
ssh mkdir + scp sources + scp payloads → remote `g++ -std=c++17 -O2 ...
gguf_loader.cpp <id>_compare.cpp` → run `<bin> <model> [token_id] <payloads>
[flags]` → parse stdout JSON → gate → write manifest/stage/build/stdout/compare/
correctness/metrics/summary. Steps before/after the boundary-specific middle are
pure template.

## What differs, and where it goes

| difference | handling |
|---|---|
| target / source / artifact_prefix / schema string / timeout | `boundaries.json` (most derivable from `id`) |
| thresholds (max_abs_diff / rmse / cosine) | `boundaries.json` `gate{}` (≈3 tiers) |
| extra source files (e.g. compare_harness.hpp) | `boundaries.json` `extra_source_files[]` |
| oracle reference selection (boundary_type/layer/tensor_kind/side/size) | `boundaries.json` `payloads[]` (order = call order) |
| call contract (token_id? staging mode? extra flags) | `boundaries.json` `needs_token_id` / `staging` / `extra_flags` |
| structured gate (name/type/dims + stats) | `boundaries.json` `gate{}` |
| **complex logic** (loop-shell 40-layer manifest; router-topk integer-set gate; sampler JSON gate; expert_ids==8 checks) | per-boundary **hook** in `tools/iq36_compare_boundaries.py` (`RESOLVERS`/`GATES` dicts) |

Estimate: ~29/34 boundaries are fully declarative; ~5 need a hook. Shared
ssh/scp/build/write helpers live once in `tools/iq36_compare_lib.py`.

## Target shape

- Runner: `python3 tools/iq36-compare.py --boundary <id>` (also `--boundary all`).
- `boundaries.json` per entry gains: `timeout_s`, `extra_source_files`,
  `needs_token_id`, `staging`, `payloads[]`, `extra_flags`, `gate{}`,
  `reference_hook`, `gate_hook` (hooks null unless needed).
- `tools/iq36_compare_boundaries.py`: only the ~5 hook functions.
- `tools/iq36_compare_lib.py`: the one copy of every shared helper.

## C++ side: adopt the harness, do NOT merge mains

Each `*_compare.cpp` carries a genuinely different kernel and stdout shape;
merging into one parameterized `main()` would become a big switch (the same
anti-pattern). Instead raise `compare_harness.hpp` adoption from 1/34 to 34/34:
each cpp `#include "compare_harness.hpp"` and keeps only its boundary compute +
emit, dropping ~120 lines of duplicated stats/JSON helpers. Optionally add
`emit_compare_json(...)` to the harness so the Python gate reads a uniform
top-level schema.

## Downstream to update with the collapse

- `tools/validate_repo.py`: today it **hardcodes the 35 driver paths**; deleting
  drivers will fail validation. Change it to derive expected `.cpp` / artifacts
  from `boundaries.json` and drop the hardcoded driver list. (This also shrinks
  the 5,065-line validator.)
- `engine/CMakeLists.txt` and `tools/iq36-ladder.py` already consume
  `boundaries.json`; no change needed.

## Parity gate before deleting old drivers (on target)

For every boundary, run the old driver and the new runner against the same
oracle bundle and diff `correctness.json` (`required_checks_passed`, comparison
stats) and `metrics.jsonl`. Require numeric/byte parity. Keep `loop-shell`,
`router-topk`, `sampler` as hooks; do not force them into pure config.

## GPU probes: the same collapse, one degree worse (2026-06-30)

The 64 `tools/intel-qwen36-gpu-*-probe.py` are the same disease as the 35 CPU
drivers, one degree worse: there is **one file per layer index**
(`gpu-resident-layer7..23-*`), and the z-correction variant is split across three
files (`-native-z-correction-`, `-gpu-f32-z-correction-`, `-gpu-q4-cpu-order-z-correction-`)
that differ only in where `z` comes from. The engine layer is already O(1) (one
`layer.cpp` + `loop.cpp`), so a per-layer-index probe file is a pure structural
leak: the layer index and the z source are **parameters**, not filenames.

Target shape (mirrors the CPU collapse above):

- Runner: `python3 tools/iq36-gpu-probe.py --layer <N> --boundary <id>
  --z-source {gpu-q4-cpu-order|native|gpu-f32}` (also `--layer all`).
- `gpu-boundaries.json` per entry: OpenCL source + SHA, kernel entry points,
  oracle `payloads[]` (boundary inputs/outputs), `gate{}` thresholds, the Q4/Q6
  packed-layout selector, and `needs_full_attention` (layers in
  `FULL_ATTENTION_LAYERS` take the full-attn path, others the linear conv path).
- Shared ssh/scp/build/OpenCL-runner helpers live once in `tools/iq36_gpu_lib.py`;
  only genuinely different kernel bodies stay per-boundary (as `.cl`, not `.py`).

This is **target-validated** (needs the PTL ssh/scp/g++/Arc-B390 env); it cannot
run on this host. Migration order, same as the CPU side:

1. Build `gpu-boundaries.json` + `iq36-gpu-probe.py` on target.
2. **Parity gate:** for each closed boundary, run the old per-layer probe and the
   new runner against the same oracle bundle; diff `correctness.json`
   (`required_checks_passed`, comparison stats). Require numeric/byte parity.
3. Only then delete the old probes and **ratchet the code-volume ceiling down**
   (`tools/intel-qwen36-code-volume-check.py --set-ceilings`).

Until then, the code-volume ratchet (now covering `*-probe.py`) **structurally
blocks a new per-layer probe** — so the next layer-23 close goes through the
runner, not a new file. This is the §0.3 guardrail finally biting where the GPU
sprawl actually happens.

## Stop trigger

Code-volume / build-time is a stop trigger (methodology ch.0 §0.3). If `tools/`,
`engine/tests/`, or `output/` resume growing one-file-per-variant, stop and
generalize before continuing. This is now enforced, not advisory:
`tools/intel-qwen36-code-volume-check.py` fails the build (and `validate_repo.py`)
on growth past the frozen ceilings — covering `*-probe.py`, `*-compare.py`, and
`engine/tests/*_compare.cpp`.
