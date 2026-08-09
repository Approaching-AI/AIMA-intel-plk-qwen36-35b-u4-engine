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
