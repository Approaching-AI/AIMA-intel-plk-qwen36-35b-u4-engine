# AIMA Intel PTL Qwen3.6 35B U4 Engine

Open-source repository for a specialized batch-size-1 OpenVINO GPU inference
engine and resident HTTP service for one locked model on the Intel PTL target:

```text
Qwen3.6-35B-A3B OpenVINO U4
```

Development first established the target, oracle, roofline, and feasibility
gate, then built a correct thin vertical slice and optimized measured
bottlenecks. It is intentionally not a generic serving runtime.

## Current Target

- Hardware: Intel PTL CLS DVT2, Core Ultra X7 358H, 64 GB class LPDDR system
- Execution target: the bound local PTL machine (hostname intentionally omitted)
- OS/kernel: Ubuntu 24.04.4 LTS, `6.17.12-061712-generic`
- GPU: Intel Arc B390 GPU through OpenCL / Level Zero stack
- NPU: Intel NPU device present through Level Zero/OpenVINO stack
- Product model path: `/home/intel/Qwen3.6-35B-A3B-ov`
- Language-model BIN SHA-256:
  `46140b595760e891d9626c5bfaffc2c998cce176d0de7f6c290af5ae1f2393a4`
- Legacy GGUF path: `/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf`
  (diagnostic only)
- Scope: batch size `1`, single model, single target machine
- Product performance: candidate-specific OpenVINO GPU specialization must
  reach at least `1.10x` untouched same-host stock OpenVINO GPU in both prefill
  and decode for every accepted bucket; see the goal and acceptance matrix

## Published v0.1.0 Results

The promoted cold/no-prefix matrix passes all `21/21` cases at output 512. It
produces the exact same `10,752/10,752` greedy tokens as the locked stock
OpenVINO GPU reference, with minimum teacher-forced top-1 agreement `1.0` and
maximum KLD `0.004836565` against the `0.005` limit.

| Prompt bucket | Candidate prefill tok/s | Candidate decode tok/s | Worst prefill LCB | Worst decode LCB | Gate |
|---:|---:|---:|---:|---:|:---:|
| 2k | 2105.22 | 51.66 | 1.479464x | 1.591514x | pass |
| 4k | 2367.89 | 50.57 | 1.568074x | 1.617308x | pass |
| 8k | 2462.30 | 47.96 | 1.654805x | 1.600109x | pass |
| 16k | 2337.37 | 45.84 | 1.648845x | 1.676622x | pass |
| 32k | 2065.28 | 38.84 | 1.608355x | 1.679497x | pass |
| 64k | 1621.72 | 30.25 | 1.600691x | 1.767016x | pass |
| 128k | 1098.78 | 21.01 | 1.813360x | 1.890407x | pass |

Throughput is the candidate median across the three prompt classes in each
bucket. Speedup is the worst paired candidate/stock one-sided 95% lower
confidence bound (LCB) in that bucket, not a ratio of unpaired best runs.

See [`BENCHMARKS.md`](BENCHMARKS.md) for the complete 21-case table,
correctness eval, context smoothness, jitter, memory, methodology, limitations,
and raw-evidence verification instructions.

## Local Experiment Setup

The repository runs directly on the Intel experiment machine. Experiment
drivers execute commands in a local shell and copy files only between local
staging paths; no network transport or separate target login is required.

Older driver flags such as `--host` and `--remote-root` remain accepted only so
recorded commands stay replayable. Their defaults are locked to `local`, and
the execution helper rejects any non-local target.

## Resident OpenAI-Compatible Service

The promoted carrier now has a resident HTTP/1.1 service with OpenAI-shaped
Models, Completions, Chat Completions, and Responses endpoints. It supports
JSON/SSE output, function tools, validated structured output, bounded
prefix-state reuse, retrievable/deletable Responses state, authentication,
readiness/metrics, memory guards, and graceful shutdown.

Users submit ordinary text and message histories at arbitrary token lengths;
the service selects an internal capacity bucket. A configurable total context
limit is enforced as `prompt_tokens + max_output_tokens`, and input is never
silently truncated. The final real HTTP smoke includes a 131072-token prompt.
Production readiness also requires a full SHA-256 pass over the locked model,
the exact promoted plugin fingerprints, the custom OpenCL configuration, and
the bound OpenVINO/GenAI runtime build identities. A same-release-family wheel
from another OpenVINO commit is rejected before model hashing or compilation.

See [`engine/service/README.md`](engine/service/README.md) for installation,
API examples, supported-field boundaries, and operations. The machine-readable
contract and release gate are
[`contracts/qwen36-openai-http-service-contract.json`](contracts/qwen36-openai-http-service-contract.json)
and
[`benchmarks/intel-qwen36-35b-a3b-gguf-q4km/http-service-acceptance-matrix.json`](benchmarks/intel-qwen36-35b-a3b-gguf-q4km/http-service-acceptance-matrix.json).
The native runtime bundle also carries checksum-locked OpenVINO/oneDNN source
postimages and a build helper that reproduces both promoted GPU plugins
byte-for-byte; see
[`doc/reference/intel-qwen36-35b-a3b-gguf-q4km/openvino-plugin-rebuild.md`](doc/reference/intel-qwen36-35b-a3b-gguf-q4km/openvino-plugin-rebuild.md).
The model is a separately supplied, hash-locked input; its recoverable history
and explicit public-release boundary are documented in
[`doc/reference/intel-qwen36-35b-a3b-gguf-q4km/locked-model-provenance-boundary.md`](doc/reference/intel-qwen36-35b-a3b-gguf-q4km/locked-model-provenance-boundary.md).

The [`v0.1.0` GitHub release](https://github.com/Approaching-AI/AIMA-intel-plk-qwen36-35b-u4-engine/releases/tag/v0.1.0)
supplies the Apache-2.0 service wheel, exact
CPython 3.12 offline wheelhouse, native x86_64 runtime/plugins with source
recipes and third-party notices, and an evidence bundle. Verify every download
against the release `SHA256SUMS` before installation. Model weights are not a
release asset.

## License and model boundary

Repository source is licensed under the
[Apache License 2.0](LICENSE). The locked model is not part of this repository,
the Python distribution, or the source license grant. Operators provide the
exact external model artifact separately; startup accepts it only when every
file matches the locked model contract. The machine-readable publication
policy is
[`contracts/qwen36-openai-http-publication-policy.json`](contracts/qwen36-openai-http-publication-policy.json).
The clean public-source boundary and intentionally excluded material are
recorded in [`PUBLICATION.md`](PUBLICATION.md).

## Read First

1. `.meta-agent/AGENT-RUNTIME.md`
2. `AGENTS.md` — then follow its "Read order", the single canonical reading list
3. `doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md` — current gate and next action

The full ordered list (spine, contracts, acceptance matrix, plan) is maintained
once, in `AGENTS.md`. This file does not keep a second copy.

## Day-0 Exit

The repo is ready to leave day-0 only when these are true:

- target contract refreshed on the live Intel host
- same-model oracle bundle found or captured
- same-host denominator recorded
- per-op/per-bucket roofline refreshed at model-real shapes
- feasibility probe passes the dominant byte-stream or matvec requirement
- resident harness adapter can load model + oracle once and run one boundary

Until then, no product-performance claim is promoted.

## Local Validation

```bash
PYTHONPATH=engine/service \
  python3.12 -m unittest discover -s engine/service/tests -v
cmake -S engine -B build/engine
cmake --build build/engine
ctest --test-dir build/engine --output-on-failure
```

Every release must pass the license, third-party/model provenance, secret and
host-detail, exact-runtime identity, and real-service acceptance gates before
its tag or binary assets are published.
